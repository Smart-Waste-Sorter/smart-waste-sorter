from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import tkinter as tk
import time
from pathlib import Path
from queue import Empty, SimpleQueue
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from src.camera import CameraManager
from src.classifier import WasteClassifier
from src.config import (
    APP_NAME,
    CATEGORIES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_REQUIRED_MATCHES,
    DEFAULT_VOICE_ENABLED,
    DEFAULT_VOICE_INCLUDE_CONFIDENCE,
    DISPLAY_NAMES,
    DB_PATH,
    MODEL_PATH,
    POST_CONFIRMATION_PAUSE_SECONDS,
)
from src.history import HistoryStore
from src.paths import data_path, resource_path
from src.sorter import recommended_bin
from src.stabilizer import Stabilizer
from src.voice import VoiceAnnouncer


class App:
    """Tkinter dashboard for live, single-item waste sorting."""

    COLORS = {
        "nav": "#14264d",
        "blue": "#2878e8",
        "blue_light": "#e8f1ff",
        "green": "#159d72",
        "green_light": "#e6f7f0",
        "orange": "#e49a19",
        "purple": "#7c67c7",
        "red": "#e65056",
        "ink": "#17233c",
        "muted": "#71809a",
        "background": "#f2f6fc",
        "white": "#ffffff",
        "border": "#e1e8f1",
    }
    CATEGORY_COLORS = {
        "Plastic": "#3486ed",
        "Paper": "#d28b15",
        "Can_Metal": "#858b99",
        "Glass": "#159d72",
        "Organic": "#659c36",
        "UNKNOWN": "#e65056",
    }
    REFRESH_MS = 33
    INFERENCE_INTERVAL_MS = 500

    def __init__(self, title: str = APP_NAME):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1280x850")
        self.root.minsize(1050, 720)
        self.root.configure(bg=self.COLORS["background"])

        self.camera = CameraManager(camera_index=0)
        self.classifier = WasteClassifier(model_path=resource_path(MODEL_PATH))
        self.voice_settings_path = data_path("settings.json")
        self.voice_enabled = DEFAULT_VOICE_ENABLED
        self.voice_volume = 0.8
        self.voice_rate = 180
        self.voice_id: str | None = None
        self.voice_include_confidence = DEFAULT_VOICE_INCLUDE_CONFIDENCE
        self.sound_warning: str | None = None
        self._sound_warnings: SimpleQueue[str] = SimpleQueue()
        self._last_announced_error: str | None = None
        self._load_voice_settings()
        self.voice_announcer = VoiceAnnouncer(
            enabled=self.voice_enabled,
            volume=self.voice_volume,
            rate=self.voice_rate,
            voice_id=self.voice_id,
            include_confidence=self.voice_include_confidence,
            on_error=self._sound_warnings.put,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self._inference_executor = ThreadPoolExecutor(max_workers=1)
        self._inference_future: Future[dict] | None = None
        self.history = HistoryStore(db_path=str(data_path(Path(DB_PATH).name)))
        self.stabilizer = Stabilizer(DEFAULT_REQUIRED_MATCHES)
        self.threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.camera_ready = False
        self.camera_checked = False
        self.model_ready = False
        self.last_inference = 0
        self._inference_pause_until = 0.0
        self.last_result = ("UNKNOWN", 0.0)
        self.last_confirmed_key: str | None = None
        self.person_detection_announced = False
        self.current_page = "Dashboard"
        self._photo: ImageTk.PhotoImage | None = None
        self.page: tk.Frame | None = None
        self.nav_buttons: dict[str, tk.Button] = {}
        self.voice_toggle: tk.Checkbutton | None = None
        self.test_sound_button: tk.Button | None = None
        self.volume_scale: tk.Scale | None = None
        self.rate_scale: tk.Scale | None = None
        self.volume_value_label: tk.Label | None = None
        self.voice_choice: ttk.Combobox | None = None
        self.voice_choice_var: tk.StringVar | None = None
        self._voice_choice_ids: dict[str, str] = {}
        self.confidence_toggle: tk.Checkbutton | None = None
        self.voice_confidence_var: tk.BooleanVar | None = None

        self._load_model()
        self._build_shell()
        self.show_page("Dashboard")
        self._open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.REFRESH_MS, self._update_frame)

    def _load_model(self) -> None:
        if not Path(self.classifier.model_path).is_file():
            self.model_error = f"Model not found: {self.classifier.model_path}"
            print(f"[MODEL] ERROR: {self.model_error}", flush=True)
            return
        try:
            self.classifier.load_model()
            self.model_ready = True
            print(
                "[MODEL] Loaded waste classifier and person detector from "
                f"{self.classifier.model_path}",
                flush=True,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.model_error = str(exc)
            print(f"[MODEL] ERROR: {self.model_error}", flush=True)

    def _build_shell(self) -> None:
        self.sidebar = tk.Frame(self.root, bg=self.COLORS["nav"], width=205)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar,
            text="♻",
            font=("Segoe UI Symbol", 32),
            bg=self.COLORS["nav"],
            fg="#31c49a",
        ).pack(pady=(24, 22))

        for name, icon in (
            ("Dashboard", "▦"),
            ("Statistics", "▥"),
            ("History", "◷"),
            ("Settings", "⚙"),
            ("Model info", "ⓘ"),
        ):
            button = tk.Button(
                self.sidebar,
                text=f"  {icon}   {name}",
                anchor="w",
                relief="flat",
                borderwidth=0,
                padx=12,
                pady=11,
                font=("Segoe UI", 10),
                bg=self.COLORS["nav"],
                fg="#c5d2e7",
                activebackground=self.COLORS["blue"],
                activeforeground="white",
                command=lambda page_name=name: self.show_page(page_name),
            )
            button.pack(fill="x", padx=8, pady=2)
            self.nav_buttons[name] = button

        tk.Label(
            self.sidebar,
            text="🌱\nCleaner\nTomorrow",
            justify="left",
            font=("Segoe UI", 10),
            bg=self.COLORS["nav"],
            fg="#9fb1cf",
        ).pack(side="bottom", anchor="w", padx=22, pady=24)

        self.main_area = tk.Frame(self.root, bg=self.COLORS["background"])
        self.main_area.pack(side="left", fill="both", expand=True)
        self.header = tk.Frame(self.main_area, bg=self.COLORS["background"])
        self.header.pack(fill="x", padx=28, pady=(22, 12))

        brand = tk.Frame(self.header, bg=self.COLORS["background"])
        brand.pack(side="left")
        tk.Label(
            brand,
            text="◉",
            font=("Segoe UI", 22, "bold"),
            bg=self.COLORS["green"],
            fg="white",
            width=3,
            height=1,
        ).pack(side="left", padx=(0, 14))
        titles = tk.Frame(brand, bg=self.COLORS["background"])
        titles.pack(side="left")
        tk.Label(
            titles,
            text="Smart waste sorter",
            font=("Segoe UI", 18, "bold"),
            bg=self.COLORS["background"],
            fg=self.COLORS["ink"],
        ).pack(anchor="w")
        tk.Label(
            titles,
            text="AI-powered recycling system",
            font=("Segoe UI", 10),
            bg=self.COLORS["background"],
            fg=self.COLORS["muted"],
        ).pack(anchor="w")

        self.state_badge = tk.Label(
            self.header,
            text="●  STARTING",
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=8,
            bg="#e9eef7",
            fg=self.COLORS["muted"],
        )
        self.state_badge.pack(side="right", pady=5)

        self.navbar = tk.Frame(self.main_area, bg="#eaf0f8")
        self.navbar.pack(fill="x", padx=28, pady=(0, 14))
        for name in self.nav_buttons:
            button = tk.Button(
                self.navbar,
                text=name,
                font=("Segoe UI", 9),
                padx=16,
                pady=8,
                relief="flat",
                borderwidth=0,
                bg="#eaf0f8",
                fg=self.COLORS["muted"],
                command=lambda page_name=name: self.show_page(page_name),
            )
            button.pack(side="left", padx=3, pady=3)

        self.page_host = tk.Frame(self.main_area, bg=self.COLORS["background"])
        self.page_host.pack(fill="both", expand=True, padx=28, pady=(0, 22))

    def _card(self, parent: tk.Widget, **kwargs: object) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=self.COLORS["white"],
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
            **kwargs,
        )

    def _label(
        self,
        parent: tk.Widget,
        text: str,
        *,
        size: int = 10,
        color: str | None = None,
        bold: bool = False,
        bg: str | None = None,
        **kwargs: object,
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            font=("Segoe UI", size, "bold" if bold else "normal"),
            fg=color or self.COLORS["ink"],
            bg=bg or self.COLORS["white"],
            **kwargs,
        )

    def show_page(self, name: str) -> None:
        self.current_page = name
        if self.page is not None:
            self.page.destroy()
        self.page = tk.Frame(self.page_host, bg=self.COLORS["background"])
        self.page.pack(fill="both", expand=True)

        for button_name, button in self.nav_buttons.items():
            selected = button_name == name
            button.configure(
                bg=self.COLORS["blue"] if selected else self.COLORS["nav"],
                fg="white" if selected else "#c5d2e7",
            )
        for button in self.navbar.winfo_children():
            selected = button.cget("text") == name
            button.configure(
                bg=self.COLORS["blue"] if selected else "#eaf0f8",
                fg="white" if selected else self.COLORS["muted"],
            )

        if name == "Dashboard":
            self._build_dashboard(self.page)
        elif name == "Statistics":
            self._build_statistics(self.page)
        elif name == "History":
            self._build_history(self.page)
        elif name == "Settings":
            self._build_settings(self.page)
        else:
            self._build_model_info(self.page)
        if name == "Dashboard":
            self._sync_camera_status()

    def _build_dashboard(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg=self.COLORS["background"])
        top.pack(fill="both", expand=True)
        top.columnconfigure(0, weight=11)
        top.columnconfigure(1, weight=9)
        top.rowconfigure(0, weight=1)

        preview_card = self._card(top)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=3)
        result_card = self._card(top)
        result_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=3)

        preview_header = tk.Frame(preview_card, bg="white")
        preview_header.pack(fill="x", padx=16, pady=(12, 8))
        self._label(preview_header, "●", color=self.COLORS["red"], size=12).pack(side="left")
        self._label(preview_header, "Live camera", bold=True).pack(side="left", padx=6)
        self.camera_status = self._label(preview_header, "Camera 0 · starting", color=self.COLORS["muted"])
        self.camera_status.pack(side="right")

        self.video_label = tk.Label(
            preview_card,
            text="Waiting for camera…",
            bg="#1c2025",
            fg="#e9edf4",
            font=("Segoe UI", 12),
            height=14,
        )
        self.video_label.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.camera_hint = self._label(
            preview_card,
            "Place one item in view. Images are not saved.",
            size=9,
            color=self.COLORS["muted"],
        )
        self.camera_hint.pack(pady=(0, 12))
        self.retry_button = tk.Button(
            preview_card,
            text="Retry camera",
            command=self._retry_camera,
            relief="flat",
            bg=self.COLORS["blue"],
            fg="white",
            padx=12,
            pady=5,
        )
        self.retry_button.pack_forget()

        self._label(result_card, "⬡  Detected category", color=self.COLORS["muted"]).pack(
            anchor="w", padx=20, pady=(18, 8)
        )
        category_row = tk.Frame(result_card, bg="white")
        category_row.pack(fill="x", padx=20, pady=(0, 8))
        self.category_value = self._label(category_row, "Waiting…", size=25, bold=True)
        self.category_value.pack(side="left", anchor="center")
        self.confidence_gauge = tk.Canvas(
            category_row,
            width=88,
            height=88,
            bg="white",
            highlightthickness=0,
        )
        self.confidence_gauge.pack(side="right")
        self._draw_confidence_gauge(0)

        confidence_header = tk.Frame(result_card, bg="white")
        confidence_header.pack(fill="x", padx=20)
        self._label(confidence_header, "Confidence", bold=True).pack(side="left")
        self.confidence_value = self._label(confidence_header, "—", bold=True, color=self.COLORS["blue"])
        self.confidence_value.pack(side="right")

        self.confidence_bar = ttk.Progressbar(result_card, maximum=100, mode="determinate")
        self.confidence_bar.pack(fill="x", padx=20, pady=(8, 4))
        self._style_progressbar()
        self.threshold_note = self._label(
            result_card,
            f"Threshold {self.threshold:.0%} · Waiting for model",
            color=self.COLORS["muted"],
            size=9,
        )
        self.threshold_note.pack(anchor="w", padx=20, pady=(0, 14))

        self.bin_panel = tk.Frame(result_card, bg=self.COLORS["blue_light"])
        self.bin_panel.pack(fill="x", padx=14, pady=(0, 14))
        self.bin_title = self._label(self.bin_panel, "Put it in", color=self.COLORS["blue"], bg=self.COLORS["blue_light"])
        self.bin_title.pack(anchor="w", padx=14, pady=(10, 1))
        self.bin_value = self._label(
            self.bin_panel,
            "Model required before sorting",
            size=13,
            bold=True,
            color=self.COLORS["blue"],
            bg=self.COLORS["blue_light"],
        )
        self.bin_value.pack(anchor="w", padx=14, pady=(0, 10))

        self.model_notice = self._label(
            result_card,
            self._model_notice_text(),
            color=self.COLORS["muted"],
            size=9,
            wraplength=350,
            justify="left",
        )
        self.model_notice.pack(anchor="w", padx=18, pady=(0, 10))

        totals_card = self._card(parent)
        totals_card.pack(fill="x", pady=(14, 0))
        totals_header = tk.Frame(totals_card, bg="white")
        totals_header.pack(fill="x", padx=15, pady=(11, 4))
        self._label(totals_header, "▤  Items sorted", size=12, bold=True).pack(side="left")
        self.total_value = self._label(totals_header, "0 total", color=self.COLORS["muted"])
        self.total_value.pack(side="right")

        self.total_bar = tk.Canvas(totals_card, height=10, bg="white", highlightthickness=0)
        self.total_bar.pack(fill="x", padx=15, pady=(2, 8))
        self.count_cards = tk.Frame(totals_card, bg="white")
        self.count_cards.pack(fill="x", padx=10, pady=(0, 8))
        self.count_values: dict[str, tk.Label] = {}
        self.count_percentages: dict[str, tk.Label] = {}
        all_categories = [*CATEGORIES, "UNKNOWN"]
        for category in all_categories:
            color = self.CATEGORY_COLORS.get(category, self.COLORS["blue"])
            card = tk.Frame(self.count_cards, bg="#f8fafd")
            card.pack(side="left", fill="both", expand=True, padx=4)
            self._label(card, f"●  {DISPLAY_NAMES.get(category, category)}", size=8, color=color, bg="#f8fafd").pack(
                anchor="w", padx=8, pady=(8, 0)
            )
            row = tk.Frame(card, bg="#f8fafd")
            row.pack(anchor="w", padx=8, pady=(2, 7))
            value = self._label(row, "0", size=16, bold=True, bg="#f8fafd")
            value.pack(side="left")
            pct = self._label(row, " 0%", size=8, color=self.COLORS["muted"], bg="#f8fafd")
            pct.pack(side="left", pady=(5, 0))
            self.count_values[category] = value
            self.count_percentages[category] = pct

        self._label(
            totals_card,
            "Below the threshold, the result shows Unknown / Please check manually.",
            color=self.COLORS["muted"],
            size=9,
        ).pack(anchor="w", padx=15, pady=(0, 11))
        self._refresh_counts()

    def _style_progressbar(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Waste.Horizontal.TProgressbar",
            troughcolor="#e0eafa",
            background=self.COLORS["blue"],
            bordercolor="#e0eafa",
            lightcolor=self.COLORS["blue"],
            darkcolor=self.COLORS["blue"],
        )
        self.confidence_bar.configure(style="Waste.Horizontal.TProgressbar")

    def _model_notice_text(self) -> str:
        if self.model_ready:
            return "Model loaded. Waiting for a stable item in the camera view."
        if hasattr(self, "model_error"):
            return f"Model could not be loaded: {self.model_error}"
        return "No trained model found at model/best.pt. Complete model training before using AI sorting."

    def _build_statistics(self, parent: tk.Frame) -> None:
        card = self._card(parent)
        card.pack(fill="both", expand=True)
        self._label(card, "Statistics", size=19, bold=True).pack(anchor="w", padx=22, pady=(20, 5))
        self._label(
            card,
            "Counts are calculated from saved classification history.",
            color=self.COLORS["muted"],
        ).pack(anchor="w", padx=22, pady=(0, 15))
        records = self.history.fetch_all()
        counts, average = self._summarize(records)
        total = sum(counts.values())
        self._label(card, f"Total items: {total}", size=14, bold=True).pack(anchor="w", padx=22, pady=8)
        self._label(card, f"Unknown items: {counts['UNKNOWN']}", size=12).pack(anchor="w", padx=22, pady=5)
        self._label(card, f"Average confidence: {average:.1%}", size=12).pack(anchor="w", padx=22, pady=5)
        for category in [*CATEGORIES, "UNKNOWN"]:
            count = counts[category]
            pct = count / total * 100 if total else 0
            self._label(
                card,
                f"{DISPLAY_NAMES.get(category, category)}:  {count}  ({pct:.1f}%)",
                size=11,
                color=self.CATEGORY_COLORS.get(category, self.COLORS["ink"]),
            ).pack(anchor="w", padx=32, pady=4)

    def _build_history(self, parent: tk.Frame) -> None:
        card = self._card(parent)
        card.pack(fill="both", expand=True)
        heading = tk.Frame(card, bg="white")
        heading.pack(fill="x", padx=18, pady=16)
        self._label(heading, "History", size=19, bold=True).pack(side="left")
        tk.Button(
            heading,
            text="Clear history",
            command=self._clear_history,
            relief="flat",
            bg="#fff0f0",
            fg=self.COLORS["red"],
            padx=12,
            pady=6,
        ).pack(side="right")
        table_frame = tk.Frame(card, bg="white")
        table_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        columns = ("timestamp", "category", "confidence", "bin")
        table = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        for col, heading_text, width in (
            ("timestamp", "Time", 180),
            ("category", "Category", 140),
            ("confidence", "Confidence", 100),
            ("bin", "Recommended bin", 240),
        ):
            table.heading(col, text=heading_text)
            table.column(col, width=width, anchor="w")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for record in self.history.fetch_all():
            table.insert(
                "",
                "end",
                values=(
                    record[0],
                    DISPLAY_NAMES.get(record[1], record[1]),
                    f"{record[2]:.1%}",
                    record[4],
                ),
            )

    def _build_settings(self, parent: tk.Frame) -> None:
        canvas = tk.Canvas(
            parent,
            bg=self.COLORS["background"],
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        card = self._card(canvas)
        card_window = canvas.create_window((0, 0), window=card, anchor="nw")

        def update_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_card(event) -> None:
            canvas.itemconfigure(card_window, width=event.width)

        card.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_card)

        self._label(card, "Settings", size=19, bold=True).pack(anchor="w", padx=22, pady=(20, 5))
        self._label(
            card,
            "Adjust the confidence threshold. The default is provisional until model evaluation.",
            color=self.COLORS["muted"],
            wraplength=700,
            justify="left",
        ).pack(anchor="w", padx=22, pady=(0, 14))

        self._label(card, "Voice announcements", size=14, bold=True).pack(
            anchor="w", padx=22, pady=(0, 8)
        )
        voice_actions = tk.Frame(card, bg="white")
        voice_actions.pack(fill="x", padx=18)
        enabled_var = tk.BooleanVar(value=self.voice_enabled)
        self.voice_toggle = tk.Checkbutton(
            voice_actions,
            text="Voice announcements",
            variable=enabled_var,
            command=lambda: self._set_voice_enabled(enabled_var.get()),
            bg="white",
            fg=self.COLORS["ink"],
            activebackground="white",
            selectcolor="white",
        )
        self.voice_toggle.pack(side="left")
        self.test_sound_button = tk.Button(
            voice_actions,
            text="Test sound",
            command=self.voice_announcer.test_sound,
            relief="flat",
            bg=self.COLORS["blue"],
            fg="white",
            padx=16,
            pady=6,
        )
        self.test_sound_button.pack(side="left", padx=12)
        self._sync_voice_controls()

        volume_var = tk.DoubleVar(value=self.voice_volume * 100)
        volume_header = tk.Frame(card, bg="white")
        volume_header.pack(fill="x", padx=22, pady=(6, 0))
        self._label(volume_header, "Volume", bold=True).pack(side="left")
        self.volume_value_label = self._label(
            volume_header,
            f"{round(self.voice_volume * 100)}%",
            color=self.COLORS["blue"],
            bold=True,
        )
        self.volume_value_label.pack(side="right")
        self.volume_scale = tk.Scale(
            card,
            from_=0,
            to=100,
            orient="horizontal",
            variable=volume_var,
            command=self._set_voice_volume,
            showvalue=False,
            sliderlength=24,
            resolution=1,
            bg="white",
            highlightthickness=0,
            troughcolor="#e0eafa",
            activebackground=self.COLORS["blue"],
        )
        self.volume_scale.pack(fill="x", padx=18)
        self.volume_scale.bind("<ButtonRelease-1>", self._save_voice_settings)
        self.volume_scale.bind("<KeyRelease>", self._save_voice_settings)

        rate_var = tk.IntVar(value=self.voice_rate)
        self._label(card, "Speech speed", bold=True).pack(anchor="w", padx=22, pady=(4, 0))
        self.rate_scale = tk.Scale(
            card,
            from_=100,
            to=250,
            resolution=10,
            orient="horizontal",
            variable=rate_var,
            command=self._set_voice_rate,
            bg="white",
            highlightthickness=0,
            troughcolor="#e0eafa",
            activebackground=self.COLORS["blue"],
        )
        self.rate_scale.pack(fill="x", padx=18)
        self.rate_scale.bind("<ButtonRelease-1>", self._save_voice_settings)
        self.rate_scale.bind("<KeyRelease>", self._save_voice_settings)
        voice_row = tk.Frame(card, bg="white")
        voice_row.pack(fill="x", padx=22, pady=(4, 2))
        self._label(voice_row, "Voice", bold=True).pack(side="left")
        self.voice_choice_var = tk.StringVar()
        self.voice_choice = ttk.Combobox(
            voice_row,
            textvariable=self.voice_choice_var,
            state="readonly",
            width=36,
        )
        self.voice_choice.pack(side="right")
        self.voice_choice.bind("<<ComboboxSelected>>", self._set_voice_choice)
        self.voice_confidence_var = tk.BooleanVar(
            value=self.voice_include_confidence
        )
        self.confidence_toggle = tk.Checkbutton(
            card,
            text="Include confidence in announcements",
            variable=self.voice_confidence_var,
            command=lambda: self._set_voice_include_confidence(
                self.voice_confidence_var.get()
            ),
            bg="white",
            fg=self.COLORS["ink"],
            activebackground="white",
            selectcolor="white",
        )
        self.confidence_toggle.pack(anchor="w", padx=22, pady=(2, 4))
        self.sound_warning_label = self._label(
            card,
            self.sound_warning or "",
            color=self.COLORS["orange"],
            wraplength=700,
            justify="left",
        )
        self.sound_warning_label.pack(anchor="w", padx=22, pady=(2, 8))

        tk.Frame(card, bg=self.COLORS["border"], height=1).pack(
            fill="x", padx=22, pady=(4, 12)
        )
        threshold_value = tk.DoubleVar(value=self.threshold * 100)
        self._label(card, "Confidence threshold", bold=True).pack(anchor="w", padx=22, pady=(8, 3))
        scale_row = tk.Frame(card, bg="white")
        scale_row.pack(fill="x", padx=18)
        scale = tk.Scale(
            scale_row,
            from_=20,
            to=95,
            resolution=5,
            orient="horizontal",
            variable=threshold_value,
            bg="white",
            fg=self.COLORS["ink"],
            highlightthickness=0,
            troughcolor="#e0eafa",
            activebackground=self.COLORS["blue"],
        )
        scale.pack(side="left", fill="x", expand=True)
        threshold_display = self._label(scale_row, f"{self.threshold:.0%}", color=self.COLORS["blue"], bold=True)
        threshold_display.pack(side="right", padx=12)
        threshold_value.trace_add(
            "write",
            lambda *_: threshold_display.configure(text=f"{threshold_value.get() / 100:.0%}"),
        )

        def save_threshold() -> None:
            self.threshold = threshold_value.get() / 100
            if hasattr(self, "threshold_note"):
                self.threshold_note.configure(text=f"Threshold {self.threshold:.0%}")
            self.stabilizer.reset()
            self.state_badge.configure(text="●  READY", bg="#e7f6ed", fg="#16824b")

        tk.Button(
            card,
            text="Apply threshold",
            command=save_threshold,
            relief="flat",
            bg=self.COLORS["blue"],
            fg="white",
            padx=16,
            pady=8,
        ).pack(anchor="w", padx=22, pady=14)

        self._label(
            card,
            "Camera: default webcam (index 0) · Image saving: off · Mode: single item",
            color=self.COLORS["muted"],
        ).pack(anchor="w", padx=22, pady=8)

    def _load_voice_settings(self) -> None:
        if not self.voice_settings_path.is_file():
            return
        try:
            with self.voice_settings_path.open(encoding="utf-8") as settings_file:
                settings = json.load(settings_file)
            if not isinstance(settings, dict):
                raise ValueError("settings file must contain a JSON object")
            enabled = settings.get("voice_enabled", self.voice_enabled)
            volume = settings.get("voice_volume", self.voice_volume)
            rate = settings.get("voice_rate", self.voice_rate)
            voice_id = settings.get("voice_id", self.voice_id)
            include_confidence = settings.get(
                "voice_include_confidence", self.voice_include_confidence
            )
            if not isinstance(enabled, bool):
                raise ValueError("voice_enabled must be true or false")
            if voice_id is not None and not isinstance(voice_id, str):
                raise ValueError("voice_id must be a string or null")
            if not isinstance(include_confidence, bool):
                raise ValueError("voice_include_confidence must be true or false")
            self.voice_enabled = enabled
            self.voice_volume = max(0.0, min(1.0, float(volume)))
            self.voice_rate = max(80, min(300, int(rate)))
            self.voice_id = voice_id
            self.voice_include_confidence = include_confidence
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.sound_warning = f"Voice settings could not be loaded: {exc}"

    def _save_voice_settings(self, _event=None) -> None:
        try:
            self.voice_settings_path.parent.mkdir(parents=True, exist_ok=True)
            with self.voice_settings_path.open("w", encoding="utf-8") as settings_file:
                json.dump(
                    {
                        "voice_enabled": self.voice_enabled,
                        "voice_volume": self.voice_volume,
                        "voice_rate": self.voice_rate,
                        "voice_id": self.voice_id,
                        "voice_include_confidence": self.voice_include_confidence,
                    },
                    settings_file,
                    indent=2,
                )
        except OSError as exc:
            self._sound_warnings.put(f"Voice settings could not be saved: {exc}")

    def _set_voice_enabled(self, enabled: bool) -> None:
        self.voice_enabled = enabled
        self.voice_announcer.set_enabled(enabled)
        self._save_voice_settings()

    def _set_voice_volume(self, value: str) -> None:
        percent = max(0.0, min(100.0, float(value)))
        self.voice_volume = percent / 100
        self.voice_announcer.set_volume(self.voice_volume)
        if self.volume_value_label is not None:
            self.volume_value_label.configure(text=f"{round(percent)}%")

    def _set_voice_rate(self, value: str) -> None:
        self.voice_rate = int(float(value))
        self.voice_announcer.set_rate(self.voice_rate)

    def _set_voice_choice(self, _event=None) -> None:
        if self.voice_choice_var is None:
            return
        voice_id = self._voice_choice_ids.get(self.voice_choice_var.get())
        self.voice_id = voice_id
        self.voice_announcer.set_voice(voice_id)
        self._save_voice_settings()

    def _set_voice_include_confidence(self, include: bool) -> None:
        self.voice_include_confidence = bool(include)
        self.voice_announcer.set_include_confidence(include)
        self._save_voice_settings()

    def _show_pending_sound_warnings(self) -> None:
        try:
            while True:
                self.sound_warning = self._sound_warnings.get_nowait()
        except Empty:
            pass
        label = getattr(self, "sound_warning_label", None)
        if label is not None and label.winfo_exists():
            label.configure(text=self.sound_warning or "")
        self._sync_voice_controls()

    def _sync_voice_controls(self) -> None:
        announcer = getattr(self, "voice_announcer", None)
        state = (
            tk.NORMAL
            if getattr(announcer, "available", False)
            else tk.DISABLED
        )
        for widget in (
            getattr(self, "voice_toggle", None),
            getattr(self, "test_sound_button", None),
            getattr(self, "volume_scale", None),
            getattr(self, "rate_scale", None),
            getattr(self, "confidence_toggle", None),
        ):
            if widget is not None and widget.winfo_exists():
                widget.configure(state=state)
        voice_choice = getattr(self, "voice_choice", None)
        voice_choice_var = getattr(self, "voice_choice_var", None)
        if voice_choice is not None and voice_choice.winfo_exists():
            voice_list = getattr(announcer, "voices", ())
            choices = {
                f"{name} ({voice_id})": voice_id
                for voice_id, name in voice_list
            }
            if choices != self._voice_choice_ids:
                self._voice_choice_ids = choices
                voice_choice.configure(values=tuple(choices))
                selected = next(
                    (
                        label
                        for label, voice_id in choices.items()
                        if voice_id == getattr(self, "voice_id", None)
                    ),
                    next(iter(choices), ""),
                )
                if voice_choice_var is not None:
                    voice_choice_var.set(selected)
            voice_choice.configure(
                state="readonly" if state == tk.NORMAL and choices else tk.DISABLED
            )

    def _build_model_info(self, parent: tk.Frame) -> None:
        card = self._card(parent)
        card.pack(fill="both", expand=True)
        self._label(card, "Model info", size=19, bold=True).pack(anchor="w", padx=22, pady=(20, 10))
        info = (
            f"Model path: {Path(self.classifier.model_path)}",
            f"Model status: {'Loaded' if self.model_ready else 'Not available'}",
            "Task: single-item waste classification",
            "Expected categories: Plastic, Paper, Can / Metal, Glass, Organic",
            "Image handling: processed locally; camera frames are not saved.",
        )
        for line in info:
            self._label(card, line, size=11, color=self.COLORS["muted"]).pack(anchor="w", padx=24, pady=6)
        self._label(card, self._model_notice_text(), color=self.COLORS["red"] if not self.model_ready else self.COLORS["green"], wraplength=700, justify="left").pack(anchor="w", padx=24, pady=12)

    def _open_camera(self) -> None:
        self.camera_ready = self.camera.open()
        self.camera_checked = True
        self._sync_camera_status()

    def _sync_camera_status(self) -> None:
        if not self.camera_ready:
            if self.camera_checked:
                self._announce_error_once(
                    "camera-unavailable",
                    "Camera error. Please check the camera.",
                )
            self.camera_status.configure(text="Camera unavailable")
            self.video_label.configure(
                image="",
                text="Camera not available\nCheck connection and permissions, then retry.",
            )
            self.state_badge.configure(text="●  CAMERA ERROR", bg="#fff0f0", fg=self.COLORS["red"])
            self.retry_button.pack(pady=(0, 8))
            return

        self.retry_button.pack_forget()
        self._last_announced_error = None
        self.camera_status.configure(text="Camera 0 · connected")
        self.state_badge.configure(
            text="●  READY" if self.model_ready else "●  MODEL NEEDED",
            bg="#e7f6ed" if self.model_ready else "#fff3dc",
            fg="#16824b" if self.model_ready else "#95610b",
        )
        self.camera_hint.configure(text="Place one item in view. Images are not saved.")

    def _retry_camera(self) -> None:
        self.camera.close()
        self.camera = CameraManager(camera_index=0)
        self._open_camera()

    def _update_frame(self) -> None:
        self._show_pending_sound_warnings()
        if self.camera_ready:
            frame = self.camera.read_frame()
            if frame is None:
                self.camera_ready = False
                self._announce_error_once(
                    "camera-signal-lost",
                    "Camera error. Please check the camera.",
                )
                if self.current_page == "Dashboard":
                    self.camera_status.configure(text="Camera signal lost")
                    self.state_badge.configure(text="●  CAMERA ERROR", bg="#fff0f0", fg=self.COLORS["red"])
                    self.video_label.configure(image="", text="Camera signal lost\nReconnect the camera, then retry.")
                    self.retry_button.pack(pady=(0, 8))
            else:
                self._last_announced_error = None
                if self.current_page == "Dashboard":
                    self._check_inference()
                    self._run_inference(frame)
                    self._draw_preview(frame)
        if self.root.winfo_exists():
            self.root.after(self.REFRESH_MS, self._update_frame)

    def _run_inference(self, frame) -> None:
        current_time = time.monotonic() * 1000
        if (
            not self.model_ready
            or self._inference_future is not None
            or current_time - self.last_inference < self.INFERENCE_INTERVAL_MS
            or time.monotonic() < self._inference_pause_until
        ):
            return
        self.last_inference = current_time

        x1, y1, x2, y2 = self._item_box(frame)
        item_image = frame[y1:y2, x1:x2].copy()
        self._inference_future = self._inference_executor.submit(
            self.classifier.predict,
            item_image,
            person_detection_frame=frame,
            person_detection_region=(x1, y1, x2, y2),
        )

    def _check_inference(self) -> None:
        future = self._inference_future
        if future is None or not future.done():
            return
        self._inference_future = None
        try:
            prediction = future.result()
            if prediction.get("blocked_reason"):
                if "person" in prediction["blocked_reason"].lower():
                    self._announce_person_detected_once()
                    self._show_scan_warning(
                        title="Person detected",
                        detail=(
                            "A person was detected in the scan area. Keep your "
                            "face out of the green box and place one waste item inside it."
                        ),
                        state="PERSON IN SCAN AREA",
                    )
                else:
                    self.person_detection_announced = False
                    self._show_scan_warning(
                        title="No waste item recognized",
                        detail=(
                            "No waste category was recognized. Place one waste "
                            "item inside the green box and try again."
                        ),
                        state="NO WASTE DETECTED",
                    )
                return
            raw_label = str(prediction["label"])
            confidence = float(prediction["confidence"])
        except (RuntimeError, KeyError, TypeError, ValueError) as exc:
            self._set_state("ERROR", "#fff0f0", self.COLORS["red"])
            self.model_notice.configure(text=f"Prediction failed: {exc}")
            self._announce_error_once(
                "prediction-error",
                "Classification error. Please try again.",
            )
            return

        self.person_detection_announced = False
        self._last_announced_error = None
        category = self._canonical_category(raw_label)
        accepted = category != "UNKNOWN" and confidence >= self.threshold
        shown_category = category if accepted else "UNKNOWN"
        self.last_result = (shown_category, confidence)
        self._render_result(category, confidence, accepted)

        if self.stabilizer.is_locked and self.stabilizer.current_label != shown_category:
            self.stabilizer.reset()
            self.last_confirmed_key = None
        confirmed = self.stabilizer.update(shown_category)
        if not confirmed:
            state = "VALIDATING" if accepted else "CHECKING · UNKNOWN"
            self._set_state(state, "#eaf1ff" if accepted else "#fff0f0", self.COLORS["blue"] if accepted else self.COLORS["red"])
            return

        self._set_state(
            "LOCKED · ITEM CONFIRMED" if accepted else "UNKNOWN · CHECK MANUALLY",
            "#ddf6e6" if accepted else "#fff0f0",
            "#16824b" if accepted else self.COLORS["red"],
        )
        if self.last_confirmed_key is None:
            if accepted:
                self._inference_pause_until = (
                    time.monotonic() + POST_CONFIRMATION_PAUSE_SECONDS
                )
            bin_label = recommended_bin(shown_category, confidence, self.threshold)
            print(
                f"[DETECTION] Confirmed {shown_category} at {confidence:.1%}; "
                f"recommended bin: {bin_label}",
                flush=True,
            )
            self.history.add_record(shown_category, confidence, bin_label)
            self._announce_locked_result(shown_category, confidence)
            self._refresh_counts()

    def _announce_person_detected_once(self) -> None:
        if self.person_detection_announced:
            return
        print(
            "[DETECTION] Person detected in the scan area; blocking waste classification.",
            flush=True,
        )
        self.voice_announcer.announce_error(
            "Person detected. Please move your face out of the green box."
        )
        self.person_detection_announced = True

    def _announce_locked_result(self, category: str, confidence: float) -> None:
        if self.last_confirmed_key is not None:
            return
        self.last_confirmed_key = category
        if category == "UNKNOWN":
            self.voice_announcer.announce_unknown()
        else:
            self.voice_announcer.announce_result(
                category,
                confidence,
                threshold=self.threshold,
            )

    def _announce_error_once(self, key: str, message: str) -> None:
        if self._last_announced_error != key:
            self.voice_announcer.announce_error(message)
            self._last_announced_error = key

    def _show_scan_warning(self, *, title: str, detail: str, state: str) -> None:
        self.stabilizer.reset()
        self.last_confirmed_key = None
        self.last_result = ("UNKNOWN", 0.0)
        self.category_value.configure(text=title, fg=self.COLORS["orange"])
        self.confidence_value.configure(text="—")
        self.confidence_bar.configure(value=0)
        self._draw_confidence_gauge(0)
        self.threshold_note.configure(text="No item was sorted.")
        self.bin_panel.configure(bg=self.COLORS["blue_light"])
        self.bin_title.configure(bg=self.COLORS["blue_light"], fg=self.COLORS["blue"])
        self.bin_value.configure(
            bg=self.COLORS["blue_light"],
            fg=self.COLORS["blue"],
            text="Place one waste item in the box",
        )
        self.model_notice.configure(text=detail)
        self._set_state(state, "#fff3dc", "#95610b")

    @staticmethod
    def _item_box(frame) -> tuple[int, int, int, int]:
        """Square in the middle of the picture where the item should be placed."""
        height, width = frame.shape[:2]
        side = int(min(width, height) * 0.48)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        return x1, y1, x1 + side, y1 + side

    def _draw_preview(self, frame) -> None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._item_box(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (44, 210, 145), 2)

        label = "Place one item here"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.55, 1)
        text_y = min(height - 10, y2 + 28)
        cv2.rectangle(frame, (x1, text_y - text_h - 6), (x1 + text_w + 12, text_y + 6), (30, 30, 30), -1)
        cv2.putText(frame, label, (x1 + 6, text_y), font, 0.55, (235, 240, 245), 1, cv2.LINE_AA)

        # Keep the camera's real shape (no stretching).
        scale = min(500 / width, 312 / height)
        size = (int(width * scale), int(height * scale))
        preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo, text="", height=0)

    def _render_result(self, category: str, confidence: float, accepted: bool) -> None:
        display_category = DISPLAY_NAMES.get(category, category)
        if not accepted and category != "UNKNOWN":
            display_category = f"{display_category} · CHECK"
        self.category_value.configure(
            text=display_category,
            fg=self.CATEGORY_COLORS.get(category, self.COLORS["red"]),
        )
        self.confidence_value.configure(text=f"{confidence:.0%}")
        self.confidence_bar.configure(value=max(0, min(100, confidence * 100)))
        self._draw_confidence_gauge(confidence)
        verdict = (
            "Passed"
            if accepted
            else "Below threshold · top prediction only; check manually"
        )
        self.threshold_note.configure(text=f"{verdict}  ·  Threshold {self.threshold:.0%}")
        bin_label = recommended_bin(category, confidence, self.threshold)
        panel_color = self.COLORS["blue_light"] if accepted else "#fff0f0"
        bin_color = self.COLORS["blue"] if accepted else self.COLORS["red"]
        self.bin_panel.configure(bg=panel_color)
        self.bin_title.configure(bg=panel_color, fg=bin_color)
        self.bin_value.configure(bg=panel_color, fg=bin_color, text=bin_label)
        if self.model_ready:
            self.model_notice.configure(text="AI runs locally. Camera frames are not saved.")

    def _draw_confidence_gauge(self, confidence: float) -> None:
        canvas = self.confidence_gauge
        canvas.delete("all")
        canvas.create_oval(7, 7, 81, 81, outline="#e5edf8", width=8)
        canvas.create_arc(
            7,
            7,
            81,
            81,
            start=90,
            extent=-max(0, min(1, confidence)) * 360,
            style="arc",
            outline=self.COLORS["blue"],
            width=8,
        )
        canvas.create_text(
            44,
            44,
            text=f"{confidence:.0%}",
            fill=self.COLORS["ink"],
            font=("Segoe UI", 12, "bold"),
        )

    @staticmethod
    def _canonical_category(label: str) -> str:
        key = "".join(char.lower() for char in label if char.isalnum())
        aliases = {
            "plastic": "Plastic",
            "paper": "Paper",
            "canmetal": "Can_Metal",
            "metal": "Can_Metal",
            "can": "Can_Metal",
            "glass": "Glass",
            "organic": "Organic",
            "organicwaste": "Organic",
        }
        return aliases.get(key, "UNKNOWN")

    def _set_state(self, text: str, background: str, foreground: str) -> None:
        self.state_badge.configure(text=f"●  {text}", bg=background, fg=foreground)

    @staticmethod
    def _summarize(records: list[tuple]) -> tuple[dict[str, int], float]:
        counts = {category: 0 for category in [*CATEGORIES, "UNKNOWN"]}
        confidences: list[float] = []
        for record in records:
            category, confidence = record[1], record[2]
            normalized = App._canonical_category(str(category))
            counts[normalized] += 1
            confidences.append(float(confidence))
        average = sum(confidences) / len(confidences) if confidences else 0.0
        return counts, average

    def _refresh_counts(self) -> None:
        if not hasattr(self, "count_values"):
            return
        counts, _ = self._summarize(self.history.fetch_all())
        total = sum(counts.values())
        self.total_value.configure(text=f"{total} total")
        self.total_bar.delete("all")
        width = max(self.total_bar.winfo_width(), 500)
        x = 0
        for category in [*CATEGORIES, "UNKNOWN"]:
            count = counts[category]
            self.count_values[category].configure(text=str(count))
            pct = count / total * 100 if total else 0
            self.count_percentages[category].configure(text=f" {pct:.0f}%")
            if total:
                segment = width * count / total
                self.total_bar.create_rectangle(
                    x,
                    1,
                    x + segment,
                    9,
                    fill=self.CATEGORY_COLORS.get(category, self.COLORS["blue"]),
                    outline="",
                )
                x += segment

    def _clear_history(self) -> None:
        self.history.clear()
        self._refresh_counts()
        self.show_page("History")

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.camera.close()
        self._inference_executor.shutdown(wait=False, cancel_futures=True)
        self.voice_announcer.close()
        self.root.destroy()