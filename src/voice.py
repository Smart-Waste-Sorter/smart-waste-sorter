from __future__ import annotations

import math
from queue import Empty, Queue
from threading import Lock, Thread
import sys
import time
from typing import Callable

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_VOICE_COOLDOWN_SECONDS,
    VOICE_CONFIDENCE_TEMPLATE,
    VOICE_DETECTION_TEMPLATE,
    VOICE_ERROR_TEMPLATE,
    VOICE_GUIDANCE_TEMPLATE,
    VOICE_NOT_RECOGNIZED_MESSAGE,
    VOICE_TEST_MESSAGE,
    VOICE_UNKNOWN_MESSAGE,
    VOICE_UNMAPPED_TEMPLATE,
    WASTE_INFO,
)


def _normalize_label(label: str) -> str:
    label_text = str(label or "")
    return "".join(character.lower() for character in label_text if character.isalnum())


def _coerce_confidence(value: object, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(coerced):
        return default
    return coerced


class VoiceAnnouncer:
    """Own one TTS engine on a worker thread and queue only the latest speech."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        volume: float = 0.8,
        rate: int = 180,
        voice_id: str | None = None,
        include_confidence: bool = True,
        cooldown_seconds: float = DEFAULT_VOICE_COOLDOWN_SECONDS,
        on_error: Callable[[str], None] | None = None,
        class_info: dict[str, dict] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.enabled = bool(enabled)
        self.volume = self._clamp_volume(volume)
        self.rate = self._clamp_rate(rate)
        self.voice_id = voice_id
        self.include_confidence = bool(include_confidence)
        cooldown_value = _coerce_confidence(cooldown_seconds, default=DEFAULT_VOICE_COOLDOWN_SECONDS)
        self.cooldown_seconds = max(0.0, cooldown_value if cooldown_value is not None else DEFAULT_VOICE_COOLDOWN_SECONDS)
        threshold_value = _coerce_confidence(
            confidence_threshold, default=DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.confidence_threshold = (
            DEFAULT_CONFIDENCE_THRESHOLD
            if threshold_value is None
            else threshold_value
        )
        self.on_error = on_error
        self.class_info = class_info if class_info is not None else WASTE_INFO
        self.available = False
        self.voices: tuple[tuple[str, str], ...] = ()
        self._voice_names = {
            _normalize_label(name): info
            for name, info in self.class_info.items()
        }
        self._messages: Queue[tuple[str, ...] | None] = Queue(maxsize=1)
        self._lock = Lock()
        self._last_announcement = float("-inf")
        self._closed = False
        self._worker = Thread(target=self._run, name="waste-voice", daemon=True)
        self._worker.start()

    @staticmethod
    def _clamp_volume(volume: float) -> float:
        return max(0.0, min(1.0, float(volume)))

    @staticmethod
    def _clamp_rate(rate: int) -> int:
        return max(80, min(300, int(rate)))

    def _resolve_threshold(self, threshold: float | None) -> float:
        resolved = (
            self.confidence_threshold
            if threshold is None
            else _coerce_confidence(threshold, default=self.confidence_threshold)
        )
        return self.confidence_threshold if resolved is None else float(resolved)

    def _is_confident_enough(self, confidence: object, threshold: float | None) -> bool:
        threshold_value = self._resolve_threshold(threshold)
        confidence_value = _coerce_confidence(confidence, default=None)
        if confidence_value is None:
            return False
        return confidence_value >= threshold_value

    def build_result_message(
        self,
        label: str,
        confidence: float,
        *,
        threshold: float | None = None,
    ) -> str:
        """Build a result sentence, never recommending a bin below threshold."""
        label_text = str(label or "").strip()
        if label_text.upper() == "UNKNOWN" or not self._is_confident_enough(
            confidence, threshold
        ):
            return VOICE_UNKNOWN_MESSAGE

        if _normalize_label(label_text) in {"other", "nonwaste"}:
            return VOICE_NOT_RECOGNIZED_MESSAGE

        info = self._voice_names.get(_normalize_label(label_text))
        if info is None:
            return VOICE_UNMAPPED_TEMPLATE.format(name=label_text or "Unknown item")

        return " ".join(
            self._build_waste_messages(
                info,
                confidence,
                include_confidence=self.include_confidence,
            )
        )

    def _build_waste_messages(
        self,
        info: dict,
        confidence: float,
        *,
        include_confidence: bool,
    ) -> tuple[str, str]:
        confidence_value = _coerce_confidence(confidence, default=0.0) or 0.0
        detection_message = (
            f"{info['display_name']} detected."
            if not include_confidence
            else VOICE_DETECTION_TEMPLATE.format(
                name=info["display_name"],
                confidence_text=VOICE_CONFIDENCE_TEMPLATE.format(
                    percent=round(
                        max(0.0, min(1.0, confidence_value)) * 100
                    )
                ),
            )
        )
        note = info.get("note", "")
        note_text = f" {note}." if note else ""
        guidance_message = VOICE_GUIDANCE_TEMPLATE.format(
            bin_name=info["bin_name"],
            recyclable_text=info.get(
                "recyclable_text",
                (
                    "Unknown recycling status"
                    if info["recyclable"] is None
                    else "Recyclable"
                    if info["recyclable"]
                    else "Not recyclable"
                ),
            ),
            note_text=note_text,
        )
        return detection_message, guidance_message

    def announce_result(
        self,
        label: str,
        confidence: float,
        *,
        threshold: float | None = None,
    ) -> None:
        self.announce_waste(label, confidence, threshold=threshold)

    def announce_waste(
        self,
        label: str,
        confidence: float,
        *,
        threshold: float | None = None,
    ) -> None:
        if str(label or "").strip().upper() == "UNKNOWN" or not self._is_confident_enough(
            confidence, threshold
        ):
            self.announce_unknown()
            return

        normalized_label = _normalize_label(str(label or "").strip())
        if normalized_label in {"other", "nonwaste"}:
            self.speak(VOICE_NOT_RECOGNIZED_MESSAGE)
            return

        info = self._voice_names.get(normalized_label)
        if info is None:
            self.speak(
                VOICE_UNMAPPED_TEMPLATE.format(name=str(label or "").strip() or "Unknown item")
            )
            return

        messages = self._build_waste_messages(
            info,
            confidence,
            include_confidence=self.include_confidence,
        )
        self._enqueue_batch(messages)

    def speak(self, message: str) -> None:
        self._enqueue(message)

    def announce_sorting_complete(self) -> None:
        """Speak only after the caller receives a real sorter-complete signal."""
        self.speak("Sorting complete. Thank you for recycling!")

    def announce_unknown(self) -> None:
        self._enqueue(VOICE_UNKNOWN_MESSAGE)

    def announce_error(self, message: str) -> None:
        self._enqueue(VOICE_ERROR_TEMPLATE.format(message=message.strip()))

    def test_sound(self) -> None:
        self._enqueue(VOICE_TEST_MESSAGE)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)
            if not self.enabled:
                self._clear_pending()

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self.volume = self._clamp_volume(volume)

    def set_rate(self, rate: int) -> None:
        with self._lock:
            self.rate = self._clamp_rate(rate)

    def set_voice(self, voice_id: str | None) -> None:
        with self._lock:
            self.voice_id = voice_id or None

    def set_include_confidence(self, include: bool) -> None:
        with self._lock:
            self.include_confidence = bool(include)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._clear_pending()
            self._messages.put_nowait(None)
        self._worker.join(timeout=1)

    def _clear_pending(self) -> None:
        while True:
            try:
                self._messages.get_nowait()
            except Empty:
                return
            else:
                self._messages.task_done()

    def _enqueue(self, message: str) -> None:
        self._enqueue_batch((message,))

    def _enqueue_batch(self, messages: tuple[str, ...]) -> None:
        if not messages:
            return
        with self._lock:
            if self._closed:
                print("[VOICE] Skipped announcement because the speech worker is closed.", flush=True)
                return
            if not self.enabled:
                print("[VOICE] Skipped announcement because voice is disabled in Settings.", flush=True)
                return
            self._clear_pending()
            self._messages.put_nowait(messages)
        print(f"[VOICE] Announcement queued: {' '.join(messages)}", flush=True)

    def _report_error(self, error: Exception) -> None:
        self.available = False
        message = (
            f"Voice announcements unavailable: {error} "
            f"(Python: {sys.executable})"
        )
        print(f"[VOICE] ERROR: {message}", flush=True)
        if self.on_error is not None:
            self.on_error(message)

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            self.voices = tuple(
                (str(voice.id), str(voice.name))
                for voice in (voices or ())
                if getattr(voice, "id", None)
            )
            with self._lock:
                installed_ids = {voice_id for voice_id, _name in self.voices}
                if self.voice_id not in installed_ids and self.voices:
                    self.voice_id = self.voices[0][0]
            self.available = True
            print(
                f"[VOICE] Speech engine initialized; voices found: {len(self.voices)}",
                flush=True,
            )
        except Exception as exc:
            self._report_error(exc)
            return

        try:
            while True:
                messages = self._messages.get()
                try:
                    if self._closed or messages is None:
                        return
                    while True:
                        with self._lock:
                            if self._closed:
                                return
                            if not self.enabled:
                                break
                            cooldown_remaining = max(
                                0.0,
                                self.cooldown_seconds
                                - (time.monotonic() - self._last_announcement),
                            )
                        if cooldown_remaining <= 0:
                            break
                        try:
                            latest_message = self._messages.get(
                                timeout=cooldown_remaining
                            )
                        except Empty:
                            continue
                        self._messages.task_done()
                        messages = latest_message
                        if self._closed or messages is None:
                            return

                    with self._lock:
                        if not self.enabled:
                            continue
                        volume = self.volume
                        rate = self.rate
                        voice_id = self.voice_id
                    engine.setProperty("volume", volume)
                    engine.setProperty("rate", rate)
                    if voice_id:
                        engine.setProperty("voice", voice_id)
                    for message in messages:
                        with self._lock:
                            if self._closed or not self.enabled:
                                break
                        engine.say(message)
                        engine.runAndWait()
                    self._last_announcement = time.monotonic()
                    print("[VOICE] Speech request completed.", flush=True)
                except Exception as exc:
                    self._report_error(exc)
                    return
                finally:
                    self._messages.task_done()
        finally:
            try:
                engine.stop()
            except Exception as exc:
                self._report_error(exc)
