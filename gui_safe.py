from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import matplotlib.pyplot as plt

import gui as legacy
from pysbagen.generators import FileSpec, GenericToneSpec


class SafeSbagenGui(legacy.SbagenGui):
    """Canonical advanced studio wrapper with single-job and Tk-thread safety."""

    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self._generation_running = False
        self._preview_running = False
        self.root.title("pysbagen Advanced Studio")

    def _set_generation_running(self, running: bool) -> None:
        self._generation_running = running
        state = "disabled" if running else "normal"
        for child in self.notebook.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, legacy.ttk.Button):
                    try:
                        widget.configure(state=state)
                    except tk.TclError:
                        pass
        if self.loaded_drg is None:
            self.generate_drg_button.configure(state="disabled")

    def _run_background(self, working_message: str, job):
        if self._generation_running:
            self._finish_error("An export is already running")
            return
        self._set_generation_running(True)
        self.status.set(working_message)

        def worker():
            try:
                result = job()
            except Exception as exc:
                self.root.after(0, self._background_failed, str(exc))
            else:
                self.root.after(0, self._background_succeeded, result)

        threading.Thread(target=worker, daemon=True).start()

    def _background_succeeded(self, result):
        self._set_generation_running(False)
        self._finish_success(result)

    def _background_failed(self, message: str):
        self._set_generation_running(False)
        self._finish_error(message)

    def run_quick(self):
        try:
            params = {
                "base": float(self.quick_base.get()),
                "beat": float(self.quick_beat.get()),
                "duration": float(self.quick_duration.get()),
                "outfile": self.quick_outfile.get(),
            }
        except ValueError as exc:
            self._finish_error(f"Invalid quick-session value: {exc}")
            return
        self._run_background(
            "Generating binaural session…",
            lambda: self.controller.generate_quick(params),
        )

    def run_advanced(self):
        try:
            params = self._advanced_params()
        except ValueError as exc:
            self._finish_error(f"Invalid advanced-session value: {exc}")
            return
        self._run_background(
            "Generating advanced session…",
            lambda: self.controller.generate_quick(params),
        )

    def run_schedule(self):
        path = self.schedule_file.get().strip()
        if not path:
            self._finish_error("Select an SBG schedule first")
            return
        try:
            outfile = self.schedule_outfile.get()
            duration = self._optional_float(self.schedule_duration)
        except ValueError as exc:
            self._finish_error(f"Invalid schedule duration: {exc}")
            return
        self._run_background(
            "Generating scheduled session…",
            lambda: self.controller.generate_schedule(path, outfile, duration),
        )

    def run_loaded_drg(self):
        if self.loaded_drg is None:
            self._finish_error,"Load an I-Doser file first"
            return
        tone_sets, schedule = self.loaded_drg
        try:
            outfile = self.schedule_outfile.get()
            duration = self._optional_float(self.schedule_duration)
        except ValueError as exc:
            self._finish_error(f"Invalid schedule duration: {exc}")
            return
        self._run_background(
            "Generating loaded I-Doser session…",
            lambda: self.controller.generate_loaded_schedule(
                tone_sets, schedule, outfile, duration
            ),
        )

    def run_tones(self):
        try:
            specs = self._tone_specs()
            duration = float(self.tone_duration.get())
            outfile = self.tone_outfile.get()
            soundscape = self.soundscape_file.get().strip() or None
            soundscape_amp = float(self.soundscape_amp.get())
        except ValueError as exc:
            self._finish_error(f"Invalid tone-builder value: {exc}")
            return
        if not specs:
            self._finish_error("Add at least one tone")
            return
        self._run_background(
            "Generating tone-builder session…",
            lambda: self.controller.generate_tones(
                specs, duration, outfile, soundscape, soundscape_amp
            ),
        )

    def preview_tones(self):
        try:
            specs = [
                GenericToneSpec(
                    freq=float(item["freq"].get()),
                    amp=float(item["amp"].get()),
                    waveform=item["waveform"].get(),
                )
                for item in self.tone_generators
            ]
            soundscape = self.soundscape_file.get().strip()
            if soundscape:
                specs.append(
                    FileSpec(
                        path=soundscape,
                        amp=float(self.soundscape_amp.get()),
                        loop=True,
                    )
                )
            duration = float(self.tone_duration.get())
        except ValueError as exc:
            self._finish_error(f"Invalid preview value: {exc}")
            return
        self._preview(specs, duration)

    def draw_pattern(self):
        try:
            frequency = float(self.visual_frequency.get())
        except ValueError as exc:
            self._finish_error(f"Invalid visualization frequency: {exc}")
            return
        n, m = legacy.viz.map_freq_to_params(frequency)
        figure = legacy.viz.generate_chladni_pattern(n, m)
        if self.pattern_canvas is not None:
            old_figure = self.pattern_canvas.figure
            self.pattern_canvas.get_tk_widget().destroy()
            plt.close(old_figure)
        self.pattern_canvas = legacy.FigureCanvasTkAgg(figure, master=self.pattern_host)
        self.pattern_canvas.draw()
        self.pattern_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status.set(f"Pattern ({n}, {m}) for {frequency:g} Hz")

    def _preview(self, specs, duration: float):
        if self._preview_running:
            self._finish_error("A preview is already playing")
            return
        self._preview_running = True
        self.status.set("Playing preview…")

        def finish(message: str, error: bool = False):
            self._preview_running = False
            self.status.set(message)
            if error:
                messagebox.showerror("pysbagen", message)

        def worker():
            try:
                self.controller.playback(specs, duration)
            except Exception as exc:
                self.root.after(0, finish, f"Preview error: {exc}", True)
            else:
                self.root.after(0, finish, "Preview complete")

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    SafeSbagenGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
