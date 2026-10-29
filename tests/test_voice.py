import sys
from threading import Event, get_ident
from types import ModuleType

import pytest

from src.config import VOICE_NOT_RECOGNIZED_MESSAGE, WASTE_INFO
from src.ui import App
from src.voice import VoiceAnnouncer, _normalize_label


def make_announcer_for_message_tests():
    announcer = VoiceAnnouncer.__new__(VoiceAnnouncer)
    announcer.class_info = WASTE_INFO
    announcer._voice_names = {
        _normalize_label(name): info for name, info in WASTE_INFO.items()
    }
    announcer.confidence_threshold = 0.6
    announcer.include_confidence = True
    return announcer


def test_sentence_builder_for_paper_with_unknown_local_acceptance():
    announcer = make_announcer_for_message_tests()

    assert announcer.build_result_message("Paper", 0.92) == (
        "Paper detected. Confidence, 92 percent. Please place this item in the "
        "paper recycling bin. Paper can be recycled when it is clean and dry. "
        "Please check whether your local service accepts it."
    )


def test_sentence_builder_for_non_recyclable_class():
    announcer = make_announcer_for_message_tests()

    assert announcer.build_result_message("Organic", 0.9) == (
        "Organic waste detected. Confidence, 90 percent. Please place this item "
        "in the organic waste bin. "
        "Organic waste can be composted when suitable facilities are available."
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            "Plastic",
            "Plastic detected. Confidence, 90 percent. Please place this item "
            "in the yellow plastic bin. Some plastics can be recycled. Please "
            "check whether your local recycling service accepts it.",
        ),
        (
            "Paper",
            "Paper detected. Confidence, 90 percent. Please place this item in "
            "the paper recycling bin. Paper can be recycled when it is clean "
            "and dry. Please check whether your local service accepts it.",
        ),
        (
            "Can_Metal",
            "Metal can detected. Confidence, 90 percent. Please place this item "
            "in the metal recycling bin. Metal cans are generally recyclable. "
            "Please empty the can before recycling and check local acceptance.",
        ),
        (
            "Glass",
            "Glass detected. Confidence, 90 percent. Please place this item in "
            "the green glass recycling bin. Glass may be recyclable. Please "
            "check whether your local service accepts it and handle broken "
            "glass carefully.",
        ),
        (
            "Organic",
            "Organic waste detected. Confidence, 90 percent. Please place this "
            "item in the organic waste bin. Organic waste can be composted when "
            "suitable facilities are available.",
        ),
    ],
)
def test_each_model_category_says_its_type_recycling_status_and_bin(label, expected):
    announcer = make_announcer_for_message_tests()

    assert announcer.build_result_message(label, 0.9) == expected


def test_unknown_low_confidence_and_unmapped_class_do_not_recommend_a_bin():
    announcer = make_announcer_for_message_tests()

    assert announcer.build_result_message("UNKNOWN", 0.99) == (
        "I am not sure what this item is. Please check it manually."
    )
    assert announcer.build_result_message("Paper", 0.4) == (
        "I am not sure what this item is. Please check it manually."
    )
    assert announcer.build_result_message("Paper", float("nan")) == (
        "I am not sure what this item is. Please check it manually."
    )
    assert announcer.build_result_message("Paper", None) == (
        "I am not sure what this item is. Please check it manually."
    )
    assert announcer.build_result_message("Textile", 0.99) == (
        "Item detected: Textile. No bin recommendation available."
    )


def test_announce_waste_queues_detection_then_guidance_only_after_threshold():
    announcer = make_announcer_for_message_tests()
    queued = []
    unknown = []
    announcer._enqueue_batch = queued.append
    announcer.announce_unknown = lambda: unknown.append(True)

    announcer.announce_waste("Can_Metal", 0.95, threshold=0.6)
    announcer.announce_waste("Paper", 0.59, threshold=0.6)

    assert queued == [
        (
            "Metal can detected. Confidence, 95 percent.",
            "Please place this item in the metal recycling bin. Metal cans are "
            "generally recyclable. Please empty the can before recycling and "
            "check local acceptance.",
        )
    ]
    assert unknown == [True]


def test_sorting_complete_message_is_available_for_confirmed_hardware_callback():
    announcer = VoiceAnnouncer.__new__(VoiceAnnouncer)
    messages = []
    announcer.speak = messages.append

    announcer.announce_sorting_complete()

    assert messages == ["Sorting complete. Thank you for recycling!"]


def test_other_waste_message_is_ready_for_a_waste_only_model():
    announcer = make_announcer_for_message_tests()

    assert announcer.build_result_message("Other", 0.9) == (
        VOICE_NOT_RECOGNIZED_MESSAGE
    )


def test_announce_locked_result_only_fires_once_until_ready_resets_key():
    class FakeAnnouncer:
        calls = []

        def announce_result(self, label, confidence, *, threshold):
            self.calls.append(("result", label, confidence, threshold))

        def announce_unknown(self):
            self.calls.append(("unknown",))

    app = App.__new__(App)
    app.last_confirmed_key = None
    app.threshold = 0.6
    app.voice_announcer = FakeAnnouncer()

    app._announce_locked_result("Paper", 0.92)
    app._announce_locked_result("Paper", 0.92)
    assert app.voice_announcer.calls == [("result", "Paper", 0.92, 0.6)]

    app.last_confirmed_key = None
    app._announce_locked_result("Paper", 0.9)
    assert len(app.voice_announcer.calls) == 2


def test_unknown_transition_speaks_only_manual_check():
    class FakeAnnouncer:
        def __init__(self):
            self.calls = []

        def announce_result(self, *args, **kwargs):
            self.calls.append(("result", args, kwargs))

        def announce_unknown(self):
            self.calls.append(("unknown",))

    app = App.__new__(App)
    app.last_confirmed_key = None
    app.threshold = 0.6
    app.voice_announcer = FakeAnnouncer()

    app._announce_locked_result("UNKNOWN", 0.25)
    app._announce_locked_result("UNKNOWN", 0.25)

    assert app.voice_announcer.calls == [("unknown",)]


def test_person_warning_is_announced_once_until_person_is_absent():
    class FakeAnnouncer:
        def __init__(self):
            self.messages = []

        def announce_error(self, message):
            self.messages.append(message)

    app = App.__new__(App)
    app.person_detection_announced = False
    app.voice_announcer = FakeAnnouncer()

    app._announce_person_detected_once()
    app._announce_person_detected_once()
    assert app.voice_announcer.messages == [
        "Person detected. Please move your face out of the green box."
    ]

    app.person_detection_announced = False
    app._announce_person_detected_once()
    assert len(app.voice_announcer.messages) == 2


def test_test_sound_uses_single_worker_and_applies_settings_before_speech(
    monkeypatch,
):
    ready = Event()
    spoken = Event()
    worker_ids = []
    settings = []
    utterances = []

    class Voice:
        id = "voice-en"
        name = "English"

    class FakeEngine:
        def getProperty(self, name):
            assert name == "voices"
            return [Voice()]

        def setProperty(self, name, value):
            worker_ids.append(get_ident())
            settings.append((name, value))

        def say(self, text):
            worker_ids.append(get_ident())
            utterances.append(text)

        def runAndWait(self):
            worker_ids.append(get_ident())
            spoken.set()

        def stop(self):
            pass

    fake_pyttsx3 = ModuleType("pyttsx3")
    fake_pyttsx3.init = FakeEngine
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    announcer = VoiceAnnouncer(volume=0.8, rate=180, on_error=lambda _error: None)
    try:
        ready.wait(0.5)
        announcer.set_volume(0.4)
        announcer.set_rate(210)
        announcer.set_voice("voice-en")
        announcer.test_sound()

        assert spoken.wait(1)
        assert announcer.available
        assert announcer.voices == (("voice-en", "English"),)
        assert utterances == [
            "Paper detected. Please put it in the paper recycling bin. Paper "
            "can be recycled when it is clean and dry. Please check whether "
            "your local service accepts it."
        ]
        assert ("volume", 0.4) in settings
        assert ("rate", 210) in settings
        assert ("voice", "voice-en") in settings
        assert len(set(worker_ids)) == 1
        assert worker_ids[0] == announcer._worker.ident
    finally:
        announcer.close()


def test_disabled_announcer_does_not_queue_sound(monkeypatch):
    initialized = Event()
    utterances = []

    class FakeEngine:
        def getProperty(self, _name):
            return []

        def say(self, text):
            utterances.append(text)

        def setProperty(self, _name, _value):
            pass

        def stop(self):
            pass

    fake_pyttsx3 = ModuleType("pyttsx3")
    fake_pyttsx3.init = lambda: (initialized.set() or FakeEngine())
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    announcer = VoiceAnnouncer(enabled=False)
    try:
        assert initialized.wait(1)

        announcer.test_sound()

        assert announcer._messages.empty()
        assert utterances == []
    finally:
        announcer.close()


def test_cooldown_defers_result_announced_after_test_sound(monkeypatch):
    spoken = []
    first_announcement = Event()
    second_announcement = Event()

    class FakeEngine:
        def getProperty(self, _name):
            return []

        def setProperty(self, _name, _value):
            pass

        def say(self, text):
            spoken.append(text)
            if len(spoken) == 1:
                first_announcement.set()
            elif len(spoken) == 3:
                second_announcement.set()

        def runAndWait(self):
            pass

        def stop(self):
            pass

    fake_pyttsx3 = ModuleType("pyttsx3")
    fake_pyttsx3.init = FakeEngine
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    announcer = VoiceAnnouncer(cooldown_seconds=0.15)
    try:
        announcer.test_sound()
        assert first_announcement.wait(1)
        announcer.announce_result("Plastic", 0.95)

        assert second_announcement.wait(1)
        announcer._messages.join()

        assert spoken == [
            "Paper detected. Please put it in the paper recycling bin. Paper can be "
            "recycled when it is clean and dry. Please check whether your local "
            "service accepts it.",
            "Plastic detected. Confidence, 95 percent.",
            "Please place this item in the yellow plastic bin. Some plastics can "
            "be recycled. Please check whether your local recycling service "
            "accepts it.",
        ]
    finally:
        announcer.close()
