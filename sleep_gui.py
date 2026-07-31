from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from pysbagen.api import render_sleep, write_audio
from pysbagen.playback import play_chunks
from pysbagen.sleep import (
    DURATION_CHOICES,
    INTENSITY_LABELS,
    PROBLEM_LABELS,
    SOUND_WORLD_LABELS,
    SleepLayers,
    SleepRequest,
    build_sleep_recipe,
    write_recipe_manifest,
)


class SleepGuideGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.page_index = 0
        self.pages: list[ttk.Frame] = []
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        root.title("PySbagen Sleep Guide")
        root.geometry("760x650")
        root.minsize(680, 560)

        ttk.Label(
            root,
            text="PySbagen Sleep Guide",
            font=("Helvetica", 20, "bold"),
        ).pack(pady=(16, 4))
        ttk.Label(
            root,
            text="Tell me what is happening tonight. PySbagen will build a matched, gradually fading audio journey.",
            wraplength=680,
            justify="center",
        ).pack(pady=(0, 12))

        self.page_host = ttk.Frame(root, padding=14)
        self.page_host.pack(fill="both", expand=True, padx=16)

        self.problem = tk.StringVar(value="racing_mind")
        self.sound_world = tk.StringVar(value="warm_ambient")
        self.intensity = tk.StringVar(value="balanced")
        self.duration = tk.StringVar(value="45")
        self.user_audio = tk.StringVar()
        self.outfile = tk.StringVar(value="sleep-journey.wav")
        self.use_recommended = tk.BooleanVar(value=True)
        self.layer_binaural = tk.BooleanVar(value=True)
        self.layer_monaural = tk.BooleanVar(value=True)
        self.layer_isochronic = tk.BooleanVar(value=False)
        self.layer_hbox = tk.BooleanVar(value=True)
        self.summary = tk.StringVar()
        self.status = tk.StringVar(value="Ready")

        self._build_problem_page()
        self._build_sound_page()
        self._build_intensity_page()
        self._build_ready_page()
        self._show_page(0)

        controls = ttk.Frame(root, padding=(16, 8))
        controls.pack(fill="x")
        self.back_button = ttk.Button(controls, text="Back", command=self.back)
        self.back_button.pack(side="left")
        self.next_button = ttk.Button(controls, text="Next", command=self.next)
        self.next_button.pack(side="right")
        ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN, anchor="w").pack(
            fill="x", side="bottom"
        )
        self._update_navigation()

    def _new_page(self, heading: str, prompt: str) -> ttk.Frame:
        page = ttk.Frame(self.page_host)
        ttk.Label(page, text=heading, font=("Helvetica", 16, "bold")).pack(
            anchor="w", pady=(4, 8)
        )
        ttk.Label(page, text=prompt, wraplength=650, justify="left").pack(
            anchor="w", pady=(0, 14)
        )
        self.pages.append(page)
        return page

    def _radio_choices(
        self,
        parent: ttk.Frame,
        variable: tk.StringVar,
        choices: dict[str, str],
    ) -> None:
        for value, label in choices.items():
            ttk.Radiobutton(parent, text=label, variable=variable, value=value).pack(
                anchor="w", pady=6
            )

    def _build_problem_page(self):
        page = self._new_page(
            "1. What is keeping you awake?",
            "Choose the description closest to what is happening tonight.",
        )
        self._radio_choices(page, self.problem, PROBLEM_LABELS)

    def _build_sound_page(self):
        page = self._new_page(
            "2. What would feel pleasant?",
            "The pleasant layer is part of the journey, not filler hiding a tone.",
        )
        self._radio_choices(page, self.sound_world, SOUND_WORLD_LABELS)
        row = ttk.Frame(page)
        row.pack(fill="x", pady=(12, 0))
        ttk.Entry(row, textvariable=self.user_audio).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row, text="Choose my audio…", command=self.choose_audio).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(
            page,
            text="Used only when ‘Use my own music or audio’ is selected.",
        ).pack(anchor="w", pady=4)

    def _build_intensity_page(self):
        page = self._new_page(
            "3. How present should the hidden layers feel?",
            "Choose by feel. You do not need to choose frequencies.",
        )
        self._radio_choices(page, self.intensity, INTENSITY_LABELS)
        ttk.Checkbutton(
            page,
            text="Use the recommended layer blend for this sleep problem",
            variable=self.use_recommended,
            command=self._toggle_custom_layers,
        ).pack(anchor="w", pady=(16, 6))
        self.layer_frame = ttk.LabelFrame(
            page,
            text="Optional layer choices",
            padding=10,
        )
        self.layer_frame.pack(fill="x")
        for text, variable in (
            ("Binaural", self.layer_binaural),
            ("Monaural", self.layer_monaural),
            ("Soft isochronic modulation", self.layer_isochronic),
            ("Harmonic Box X-style layer", self.layer_hbox),
        ):
            ttk.Checkbutton(self.layer_frame, text=text, variable=variable).pack(
                anchor="w", pady=2
            )
        self._toggle_custom_layers()

    def _build_ready_page(self):
        page = self._new_page(
            "4. How long should it stay with you?",
            "The journey descends, enters a quieter support period, and slowly fades away.",
        )
        duration_row = ttk.Frame(page)
        duration_row.pack(fill="x", pady=6)
        ttk.Label(duration_row, text="Journey length").pack(side="left")
        ttk.Combobox(
            duration_row,
            textvariable=self.duration,
            values=tuple(str(value) for value in DURATION_CHOICES),
            state="readonly",
            width=8,
        ).pack(side="left", padx=8)
        ttk.Label(duration_row, text="minutes").pack(side="left")

        ttk.Separator(page).pack(fill="x", pady=14)
        ttk.Label(
            page,
            textvariable=self.summary,
            wraplength=650,
            justify="left",
        ).pack(anchor="w", pady=6)
        ttk.Label(
            page,
            text="Keep the volume comfortable. Stop if the audio causes pain, dizziness, agitation, or other unwanted effects.",
            wraplength=650,
        ).pack(anchor="w", pady=(6, 14))

        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=4)
        self.play_button = ttk.Button(
            actions,
            text="Put on headphones and start",
            command=self.start_playback,
        )
        self.play_button.pack(side="left")
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self.stop_playback,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=8)

        save = ttk.LabelFrame(page, text="Or save the journey", padding=8)
        save.pack(fill="x", pady=(14, 0))
        row = ttk.Frame(save)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.outfile).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row, text="Choose output…", command=self.choose_output).pack(
            side="left", padx=8
        )
        ttk.Button(
            save,
            text="Generate audio file",
            command=self.save_audio,
        ).pack(anchor="e", pady=(8, 0))

    def _toggle_custom_layers(self):
        state = "disabled" if self.use_recommended.get() else "normal"
        for child in self.layer_frame.winfo_children():
            child.configure(state=state)

    def choose_audio(self):
        path = filedialog.askopenfilename(
            title="Choose music or audio",
            filetypes=(("Audio files", "*.*"),),
        )
        if path:
            self.user_audio.set(path)
            self.sound_world.set("user_audio")

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Save sleep journey",
            defaultextension=".wav",
            filetypes=(("WAV audio", "*.wav"), ("All files", "*.*")),
        )
        if path:
            self.outfile.set(path)

    def _request(self) -> SleepRequest:
        layers = None
        if not self.use_recommended.get():
            layers = SleepLayers(
                binaural=self.layer_binaural.get(),
                monaural=self.layer_monaural.get(),
                isochronic=self.layer_isochronic.get(),
                harmonic_box=self.layer_hbox.get(),
            )
        request = SleepRequest(
            problem=self.problem.get(),
            sound_world=self.sound_world.get(),
            intensity=self.intensity.get(),
            duration_minutes=float(self.duration.get()),
            user_audio=(self.user_audio.get().strip() or None)
            if self.sound_world.get() == "user_audio"
            else None,
            layers=layers,
        )
        request.validate()
        return request

    def _refresh_summary(self):
        try:
            recipe = build_sleep_recipe(self._request())
        except (OSError, ValueError) as exc:
            self.summary.set(str(exc))
            return
        layer_text = (
            ", ".join(recipe.request.layers.enabled_names())
            if recipe.request.layers
            else "none"
        )
        self.summary.set(
            f"Matched journey: {recipe.name}\n{recipe.description}\n"
            f"Descent: {recipe.descent_seconds / 60:.0f} min · Quiet support: {recipe.support_seconds / 60:.0f} min\n"
            f"Underlying blend: {layer_text}"
        )

    def _show_page(self, index: int):
        for page in self.pages:
            page.pack_forget()
        self.page_index = index
        self.pages[index].pack(fill="both", expand=True)
        if index == len(self.pages) - 1:
            self._refresh_summary()
        self._update_navigation()

    def _update_navigation(self):
        if not hasattr(self, "back_button"):
            return
        self.back_button.configure(
            state="disabled" if self.page_index == 0 else "normal"
        )
        if self.page_index == len(self.pages) - 1:
            self.next_button.configure(text="Start over", command=self.start_over)
        else:
            self.next_button.configure(text="Next", command=self.next)

    def next(self):
        if (
            self.page_index == 1
            and self.sound_world.get() == "user_audio"
            and not self.user_audio.get().strip()
        ):
            messagebox.showerror(
                "PySbagen Sleep Guide",
                "Choose the music or audio you want to use.",
            )
            return
        self._show_page(min(self.page_index + 1, len(self.pages) - 1))

    def back(self):
        self._show_page(max(self.page_index - 1, 0))

    def start_over(self):
        self.stop_playback()
        self._show_page(0)

    def _set_running(self, running: bool, status: str):
        self.status.set(status)
        self.play_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def start_playback(self):
        try:
            request = self._request()
        except (OSError, ValueError) as exc:
            messagebox.showerror("PySbagen Sleep Guide", str(exc))
            return
        self.stop_event.clear()
        self._set_running(True, "Playing your matched sleep journey…")

        def worker():
            try:
                play_chunks(render_sleep(request), stop_event=self.stop_event)
            except Exception as exc:
                self.root.after(0, self._playback_failed, str(exc))
            else:
                self.root.after(0, self._playback_finished)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def stop_playback(self):
        self.stop_event.set()
        if hasattr(self, "stop_button"):
            self.status.set("Stopping…")
            self.stop_button.configure(state="disabled")

    def _playback_finished(self):
        status = "Stopped" if self.stop_event.is_set() else "Journey complete"
        self._set_running(False, status)

    def _playback_failed(self, message: str):
        self._set_running(False, f"Playback error: {message}")
        messagebox.showerror("PySbagen Sleep Guide", message)

    def save_audio(self):
        try:
            request = self._request()
            recipe = build_sleep_recipe(request)
            outfile = self.outfile.get().strip()
            if not outfile:
                raise ValueError("Choose an output file")
        except (OSError, ValueError) as exc:
            messagebox.showerror("PySbagen Sleep Guide", str(exc))
            return
        self.status.set("Generating sleep journey…")

        def worker():
            try:
                result = write_audio(render_sleep(request), outfile)
                manifest = write_recipe_manifest(recipe, result.outfile)
            except Exception as exc:
                self.root.after(0, self._save_failed, str(exc))
            else:
                self.root.after(0, self._save_finished, result.outfile, manifest)

        threading.Thread(target=worker, daemon=True).start()

    def _save_finished(self, outfile: Path, manifest: Path):
        self.status.set(f"Saved {outfile}")
        messagebox.showinfo(
            "Sleep journey saved",
            f"Audio: {outfile}\nRecipe: {manifest}",
        )

    def _save_failed(self, message: str):
        self.status.set(f"Generation error: {message}")
        messagebox.showerror("PySbagen Sleep Guide", message)


def main():
    root = tk.Tk()
    SleepGuideGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
