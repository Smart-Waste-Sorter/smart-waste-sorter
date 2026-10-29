from queue import SimpleQueue

from src.config import DEFAULT_VOICE_ENABLED, POST_CONFIRMATION_PAUSE_SECONDS
from src.ui import App


class _FakeRoot:
    def __init__(self):
        self.callbacks = []

    def winfo_exists(self):
        return True

    def after(self, delay, callback):
        self.callbacks.append((delay, callback))


class _FakeCamera:
    def __init__(self, frame):
        self.frame = frame
        self.read_count = 0

    def read_frame(self):
        self.read_count += 1
        return self.frame


def make_frame_updater(page):
    app = App.__new__(App)
    app.root = _FakeRoot()
    app._sound_warnings = SimpleQueue()
    app.sound_warning = None
    app.camera = _FakeCamera(object())
    app.camera_ready = True
    app.current_page = page
    calls = []
    app._check_inference = lambda: calls.append("check")
    app._run_inference = lambda frame: calls.append("infer")
    app._draw_preview = lambda frame: calls.append("draw")
    return app, calls


def test_canonical_category_maps_model_labels_to_sorting_categories():
    assert App._canonical_category("Can/Metal") == "Can_Metal"
    assert App._canonical_category("metal") == "Can_Metal"
    assert App._canonical_category("Organic waste") == "Organic"
    assert App._canonical_category("unrecognized class") == "UNKNOWN"


def test_camera_keeps_running_without_touching_hidden_dashboard_widgets():
    app, calls = make_frame_updater("Statistics")

    app._update_frame()

    assert app.camera.read_count == 1
    assert calls == []
    assert app.root.callbacks == [(App.REFRESH_MS, app._update_frame)]


def test_camera_preview_and_inference_resume_on_dashboard():
    app, calls = make_frame_updater("Dashboard")

    app._update_frame()

    assert calls == ["check", "infer", "draw"]
    assert app.root.callbacks == [(App.REFRESH_MS, app._update_frame)]


def test_camera_error_is_not_announced_before_first_camera_attempt():
    class _Widget:
        def configure(self, **_kwargs):
            pass

        def pack(self, **_kwargs):
            pass

    app = App.__new__(App)
    app.camera_ready = False
    app.camera_checked = False
    app.camera_status = _Widget()
    app.video_label = _Widget()
    app.state_badge = _Widget()
    app.retry_button = _Widget()
    announced = []
    app._announce_error_once = lambda key, message: announced.append((key, message))

    app._sync_camera_status()

    assert announced == []

    app.camera_checked = True
    app._sync_camera_status()

    assert announced == [
        ("camera-unavailable", "Camera error. Please check the camera.")
    ]


def test_inference_waits_during_post_confirmation_pause(monkeypatch):
    class _Executor:
        def __init__(self):
            self.submissions = 0

        def submit(self, *_args, **_kwargs):
            self.submissions += 1

    app = App.__new__(App)
    app.model_ready = True
    app._inference_future = None
    app.last_inference = 0
    app._inference_executor = _Executor()
    app._inference_pause_until = 100.0 + POST_CONFIRMATION_PAUSE_SECONDS
    monkeypatch.setattr("src.ui.time.monotonic", lambda: 100.0)

    app._run_inference(object())

    assert app._inference_executor.submissions == 0


def test_history_summary_counts_unknown_and_confidence():
    records = [
        ("2026-10-06T09:00:00", "Plastic", 0.9, None, "Plastic / Recyclable"),
        ("2026-10-06T09:01:00", "UNKNOWN", 0.2, None, "General Waste / check manually"),
        ("2026-10-06T09:02:00", "Can_Metal", 0.8, None, "Metal / Recyclable"),
    ]

    counts, average = App._summarize(records)

    assert counts["Plastic"] == 1
    assert counts["Can_Metal"] == 1
    assert counts["UNKNOWN"] == 1
    assert abs(average - 0.6333333333333333) < 1e-9


def test_volume_setting_updates_engine_value_and_percent_label():
    class _VoiceAnnouncer:
        volume = None

        def set_volume(self, value):
            self.volume = value

    class _ValueLabel:
        text = None

        def configure(self, *, text):
            self.text = text

    app = App.__new__(App)
    app.voice_announcer = _VoiceAnnouncer()
    app.volume_value_label = _ValueLabel()

    app._set_voice_volume("80")

    assert app.voice_volume == 0.8
    assert app.voice_announcer.volume == 0.8
    assert app.volume_value_label.text == "80%"


def test_low_confidence_shows_candidate_but_keeps_bin_unknown():
    class _Widget:
        def __init__(self):
            self.values = {}

        def configure(self, **kwargs):
            self.values.update(kwargs)

    class _Canvas(_Widget):
        def delete(self, *_args):
            pass

        def create_oval(self, *_args, **_kwargs):
            pass

        def create_arc(self, *_args, **_kwargs):
            pass

        def create_text(self, *_args, **_kwargs):
            pass

    app = App.__new__(App)
    app.threshold = 0.6
    app.model_ready = True
    app.COLORS = App.COLORS
    app.category_value = _Widget()
    app.confidence_value = _Widget()
    app.confidence_bar = _Widget()
    app.confidence_gauge = _Canvas()
    app.threshold_note = _Widget()
    app.bin_panel = _Widget()
    app.bin_title = _Widget()
    app.bin_value = _Widget()
    app.model_notice = _Widget()

    app._render_result("Paper", 0.55, False)

    assert app.category_value.values["text"] == "Paper · CHECK"
    assert "top prediction only" in app.threshold_note.values["text"]
    assert app.bin_value.values["text"].startswith("Unknown")


def test_voice_controls_follow_tts_availability():
    class _Widget:
        def __init__(self):
            self.state = None

        def winfo_exists(self):
            return True

        def configure(self, *, state):
            self.state = state

    class _VoiceAnnouncer:
        available = False

    app = App.__new__(App)
    app.voice_announcer = _VoiceAnnouncer()
    widgets = [_Widget() for _ in range(4)]
    (
        app.voice_toggle,
        app.test_sound_button,
        app.volume_scale,
        app.rate_scale,
    ) = widgets

    app._sync_voice_controls()
    assert all(widget.state == "disabled" for widget in widgets)

    app.voice_announcer.available = True
    app._sync_voice_controls()
    assert all(widget.state == "normal" for widget in widgets)


def test_voice_announcements_are_enabled_by_default(tmp_path):
    app = App.__new__(App)
    app.voice_settings_path = tmp_path / "settings.json"
    app.voice_enabled = DEFAULT_VOICE_ENABLED

    app._load_voice_settings()

    assert app.voice_enabled is True
