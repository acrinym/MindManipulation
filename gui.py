from __future__ import annotations

import io
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

import drg_decoder
import visualization as viz
from pysbagen.api import build_quick_specs, render_schedule, render_specs, write_audio
from pysbagen.generators import FileSpec, GenericToneSpec, IsochronicSpec
from pysbagen.mixer import build_session_generator
from pysbagen.parser import parse_sbg_from_string

try:
    import pyaudio
except ImportError:  # Audio export remains available without live playback extras.
    pyaudio = None


class SbagenController:
    def generate_quick(self, params: dict):
        specs = build_quick_specs(
            base=params.get("base"),
            beat=params.get("beat"),
            isochronic=params.get("isochronic"),
            harmonic_box=params.get("harmonic_box"),
            noise=params.get("noise"),
            noise_kind=params.get("noise_kind", "white"),
            music=params.get("music"),
            music_amp=params.get("music_amp", 100.0),
            loop_music=params.get("loop_music", False),
        )
        return write_audio(render_specs(specs, float(params["duration"])), params["outfile"])

    def generate_schedule(self, path: str, outfile: str, duration: float | None = None):
        return write_audio(render_schedule(path, duration), outfile)

    def generate_loaded_schedule(self, tone_sets, schedule, outfile: str, duration: float | None):
        return write_audio(build_session_generator(tone_sets, schedule, duration), outfile)

    def generate_tones(
        self,
        specs: list[dict],
        duration: float,
        outfile: str,
        soundscape: str | None = None,
        soundscape_amp: float = 50.0,
        loop_soundscape: bool = True,
    ):
        generators = [
            GenericToneSpec(
                freq=float(spec["freq"]),
                amp=float(spec["amp"]),
                waveform=spec["waveform"],
            )
            for spec in specs
        ]
        if soundscape:
            generators.append(
                FileSpec(
                    path=soundscape,
                    amp=float(soundscape_amp),
                    loop=loop_soundscape,
                )
            )
        return write_audio(render_specs(generators, duration), outfile)

    def playback(self, specs, duration: float, on_chunk=None):
        if pyaudio is None:
            raise RuntimeError("Live playback requires the GUI extra: pip install 'pysbagen[gui]'")
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paFloat32, channels=2, rate=44100, output=True)
        try:
            for chunk, info in render_specs(specs, duration):
                stream.write(np.asarray(chunk, dtype=np.float32).tobytes())
                if on_chunk:
                    on_chunk(info)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


class SbagenGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.controller = SbagenController()
        self.tone_generators: list[dict] = []
        self.loaded_drg = None
        self.drg_photo = None
        self.pattern_canvas = None

        root.title("pysbagen Session Builder")
        root.geometry("820x720")
        self._build_menu()

        ttk.Label(root, text="pysbagen Session Builder", font=("Helvetica", 17, "bold")).pack(pady=(10, 2))
        ttk.Label(
            root,
            text="Build and export local binaural, isochronic, harmonic-box, noise, music, and SBG sessions.",
        ).pack(pady=(0, 8))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(padx=10, pady=5, fill="both", expand=True)
        self._build_quick_tab()
        self._build_schedule_tab()
        self._build_advanced_tab()
        self._build_tone_tab()
        self._build_visualization_tab()

        self.status = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open I-Doser file…", command=self.open_drg_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

    def _tab(self, title: str):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=title)
        return tab

    @staticmethod
    def _entry(parent, label: str, default: str = "", width: int = 24):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=24).pack(side="left")
        entry = ttk.Entry(row, width=width)
        entry.insert(0, default)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _path_row(self, parent, label: str, save: bool, filetypes):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=24).pack(side="left")
        variable = tk.StringVar()
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

        def browse():
            chooser = filedialog.asksaveasfilename if save else filedialog.askopenfilename
            selected = chooser(filetypes=filetypes)
            if selected:
                variable.set(selected)

        ttk.Button(row, text="Browse…", command=browse).pack(side="left", padx=(5, 0))
        return variable

    def _build_quick_tab(self):
        tab = self._tab("Quick Generate")
        self.quick_base = self._entry(tab, "Base frequency (Hz)", "200")
        self.quick_beat = self._entry(tab, "Beat frequency (Hz)", "10")
        self.quick_duration = self._entry(tab, "Duration (seconds)", "60")
        self.quick_outfile = self._path_row(
            tab,
            "Output WAV",
            True,
            (("WAV audio", "*.wav"), ("All files", "*.*")),
        )
        self.quick_outfile.set("session.wav")
        ttk.Button(tab, text="Generate binaural session", command=self.run_quick).pack(pady=10)

    def _build_schedule_tab(self):
        tab = self._tab("Schedule / I-Doser")
        self.schedule_file = self._path_row(
            tab,
            "SBG schedule",
            False,
            (("SBG schedules", "*.sbg"), ("All files", "*.*")),
        )
        self.schedule_duration = self._entry(tab, "Override duration (optional)", "")
        self.schedule_outfile = self._path_row(
            tab,
            "Output WAV",
            True,
            (("WAV audio", "*.wav"), ("All files", "*.*")),
        )
        self.schedule_outfile.set("scheduled-session.wav")
        actions = ttk.Frame(tab)
        actions.pack(pady=8)
        ttk.Button(actions, text="Generate selected SBG", command=self.run_schedule).pack(side="left", padx=4)
        self.generate_drg_button = ttk.Button(
            actions,
            text="Generate loaded I-Doser",
            command=self.run_loaded_drg,
            state="disabled",
        )
        self.generate_drg_button.pack(side="left", padx=4)
        self.drg_label = ttk.Label(tab, text="No I-Doser file loaded.")
        self.drg_label.pack(pady=4)
        self.drg_image_label = ttk.Label(tab)
        self.drg_image_label.pack(pady=4)

    def _build_advanced_tab(self):
        tab = self._tab("Advanced")
        self.advanced_duration = self._entry(tab, "Duration (seconds)", "60")
        self.iso_freq = self._entry(tab, "Isochronic frequency", "200")
        self.iso_beat = self._entry(tab, "Isochronic beat", "10")
        self.hbox_base = self._entry(tab, "Harmonic-box base", "180")
        self.hbox_diff = self._entry(tab, "Harmonic-box difference", "5")
        self.hbox_mod = self._entry(tab, "Harmonic-box modulation", "8")
        self.noise_amp = self._entry(tab, "Noise amplitude (%)", "0")
        self.noise_kind = tk.StringVar(value="white")
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Noise kind", width=24).pack(side="left")
        ttk.Combobox(row, textvariable=self.noise_kind, values=("white", "pink"), state="readonly").pack(side="left")
        self.music_file = self._path_row(
            tab,
            "Background audio",
            False,
            (("Audio", "*.wav *.ogg *.flac *.mp3"), ("All files", "*.*")),
        )
        self.music_amp = self._entry(tab, "Background volume (%)", "50")
        self.loop_music = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Loop background audio to session length", variable=self.loop_music).pack(anchor="w", padx=24)
        self.advanced_outfile = self._path_row(
            tab,
            "Output WAV",
            True,
            (("WAV audio", "*.wav"), ("All files", "*.*")),
        )
        self.advanced_outfile.set("advanced-session.wav")
        ttk.Button(tab, text="Generate advanced session", command=self.run_advanced).pack(pady=10)

    def _build_tone_tab(self):
        tab = self._tab("Tone Builder")
        controls = ttk.Frame(tab)
        controls.pack(fill="x")
        self.tone_duration = self._entry(controls, "Duration (seconds)", "10")
        self.tone_outfile = self._path_row(
            controls,
            "Output WAV",
            True,
            (("WAV audio", "*.wav"), ("All files", "*.*")),
        )
        self.tone_outfile.set("tone-builder.wav")
        self.soundscape_file = self._path_row(
            controls,
            "Soundscape",
            False,
            (("Audio", "*.wav *.ogg *.flac *.mp3"), ("All files", "*.*")),
        )
        self.soundscape_amp = self._entry(controls, "Soundscape volume (%)", "50")

        self.tone_rows = ttk.Frame(tab)
        self.tone_rows.pack(fill="both", expand=True, pady=8)
        buttons = ttk.Frame(tab)
        buttons.pack()
        ttk.Button(buttons, text="Add tone", command=self.add_tone_generator).pack(side="left", padx=4)
        ttk.Button(buttons, text="Export tone session", command=self.run_tones).pack(side="left", padx=4)
        ttk.Button(buttons, text="Preview tone session", command=self.preview_tones).pack(side="left", padx=4)
        self.add_tone_generator()

    def _build_visualization_tab(self):
        tab = self._tab("Visualization")
        self.visual_frequency = self._entry(tab, "Frequency (Hz)", "200")
        self.visual_beat = self._entry(tab, "Beat (Hz)", "10")
        self.visual_duration = self._entry(tab, "Preview seconds", "10")
        actions = ttk.Frame(tab)
        actions.pack(pady=5)
        ttk.Button(actions, text="Draw Chladni pattern", command=self.draw_pattern).pack(side="left", padx=4)
        ttk.Button(actions, text="Play isochronic preview", command=self.preview_visualization).pack(side="left", padx=4)
        self.pattern_host = ttk.Frame(tab)
        self.pattern_host.pack(fill="both", expand=True)

    def _run_background(self, working_message: str, job):
        self.status.set(working_message)

        def worker():
            try:
                result = job()
            except Exception as exc:
                self.root.after(0, self._finish_error, str(exc))
            else:
                self.root.after(0, self._finish_success, result)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_success(self, result):
        message = f"Wrote {result.duration:.2f}s to {result.outfile}"
        self.status.set(message)
        messagebox.showinfo("Generation complete", message)

    def _finish_error(self, message: str):
        self.status.set(f"Error: {message}")
        messagebox.showerror("pysbagen", message)

    @staticmethod
    def _optional_float(entry: ttk.Entry):
        value = entry.get().strip()
        return float(value) if value else None

    def run_quick(self):
        self._run_background(
            "Generating binaural session…",
            lambda: self.controller.generate_quick(
                {
                    "base": float(self.quick_base.get()),
                    "beat": float(self.quick_beat.get()),
                    "duration": float(self.quick_duration.get()),
                    "outfile": self.quick_outfile.get(),
                }
            ),
        )

    def _advanced_params(self):
        music = self.music_file.get().strip() or None
        return {
            "duration": float(self.advanced_duration.get()),
            "outfile": self.advanced_outfile.get(),
            "isochronic": (float(self.iso_freq.get()), float(self.iso_beat.get())),
            "harmonic_box": (
                float(self.hbox_base.get()),
                float(self.hbox_diff.get()),
                float(self.hbox_mod.get()),
            ),
            "noise": float(self.noise_amp.get()) if float(self.noise_amp.get()) > 0 else None,
            "noise_kind": self.noise_kind.get(),
            "music": music,
            "music_amp": float(self.music_amp.get()),
            "loop_music": self.loop_music.get(),
        }

    def run_advanced(self):
        self._run_background(
            "Generating advanced session…",
            lambda: self.controller.generate_quick(self._advanced_params()),
        )

    def run_schedule(self):
        path = self.schedule_file.get().strip()
        if not path:
            self._finish_error("Select an SBG schedule first")
            return
        self._run_background(
            "Generating scheduled session…",
            lambda: self.controller.generate_schedule(
                path,
                self.schedule_outfile.get(),
                self._optional_float(self.schedule_duration),
            ),
        )

    def run_loaded_drg(self):
        if self.loaded_drg is None:
            self._finish_error("Load an I-Doser file first")
            return
        tone_sets, schedule = self.loaded_drg
        self._run_background(
            "Generating loaded I-Doser session…",
            lambda: self.controller.generate_loaded_schedule(
                tone_sets,
                schedule,
                self.schedule_outfile.get(),
                self._optional_float(self.schedule_duration),
            ),
        )

    def open_drg_file(self):
        path = filedialog.askopenfilename(
            title="Select an I-Doser file",
            filetypes=(("I-Doser files", "*.drg"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            sbg_source, image_data = drg_decoder.decode_drg(path)
            self.loaded_drg = parse_sbg_from_string(sbg_source, base_dir=Path(path).parent)
            self.generate_drg_button.config(state="normal")
            self.drg_label.config(text=f"Loaded: {Path(path).name}")
            if image_data:
                image = Image.open(io.BytesIO(image_data))
                image.thumbnail((320, 240))
                self.drg_photo = ImageTk.PhotoImage(image)
                self.drg_image_label.config(image=self.drg_photo)
            self.status.set(f"Loaded I-Doser schedule: {Path(path).name}")
        except Exception as exc:
            self._finish_error(f"Could not decode I-Doser file: {exc}")

    def add_tone_generator(self):
        row = ttk.LabelFrame(self.tone_rows, text=f"Tone {len(self.tone_generators) + 1}", padding=6)
        row.pack(fill="x", pady=3)
        frequency = self._entry(row, "Frequency (Hz)", "200")
        amplitude = self._entry(row, "Amplitude (%)", "50")
        waveform = tk.StringVar(value="sine")
        selector = ttk.Combobox(
            row,
            textvariable=waveform,
            values=("sine", "square", "triangle", "sawtooth"),
            state="readonly",
        )
        selector.pack(anchor="w", padx=24, pady=2)
        record = {"frame": row, "freq": frequency, "amp": amplitude, "waveform": waveform}
        ttk.Button(row, text="Remove", command=lambda: self.remove_tone_generator(record)).pack(anchor="e")
        self.tone_generators.append(record)

    def remove_tone_generator(self, record):
        record["frame"].destroy()
        self.tone_generators.remove(record)

    def _tone_specs(self):
        return [
            {
                "freq": float(item["freq"].get()),
                "amp": float(item["amp"].get()),
                "waveform": item["waveform"].get(),
            }
            for item in self.tone_generators
        ]

    def run_tones(self):
        specs = self._tone_specs()
        if not specs:
            self._finish_error("Add at least one tone")
            return
        soundscape = self.soundscape_file.get().strip() or None
        self._run_background(
            "Generating tone-builder session…",
            lambda: self.controller.generate_tones(
                specs,
                float(self.tone_duration.get()),
                self.tone_outfile.get(),
                soundscape,
                float(self.soundscape_amp.get()),
            ),
        )

    def preview_tones(self):
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
            specs.append(FileSpec(path=soundscape, amp=float(self.soundscape_amp.get()), loop=True))
        self._preview(specs, float(self.tone_duration.get()))

    def draw_pattern(self):
        frequency = float(self.visual_frequency.get())
        n, m = viz.map_freq_to_params(frequency)
        figure = viz.generate_chladni_pattern(n, m)
        if self.pattern_canvas is not None:
            self.pattern_canvas.get_tk_widget().destroy()
        self.pattern_canvas = FigureCanvasTkAgg(figure, master=self.pattern_host)
        self.pattern_canvas.draw()
        self.pattern_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status.set(f"Pattern ({n}, {m}) for {frequency:g} Hz")

    def preview_visualization(self):
        self.draw_pattern()
        spec = IsochronicSpec(freq=float(self.visual_frequency.get()), beat=float(self.visual_beat.get()))
        self._preview([spec], float(self.visual_duration.get()))

    def _preview(self, specs, duration: float):
        self.status.set("Playing preview…")

        def worker():
            try:
                self.controller.playback(specs, duration)
            except Exception as exc:
                self.root.after(0, self._finish_error, str(exc))
            else:
                self.root.after(0, self.status.set, "Preview complete")

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    SbagenGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
