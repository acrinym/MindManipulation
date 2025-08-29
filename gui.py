import tkinter as tk
from tkinter import ttk, messagebox
import argparse
import sbagen
import soundfile as sf
import numpy as np

from pydub import AudioSegment

class SbagenController:
    def generate(self, params):
        try:
            if params.get("ffmpeg_path"):
                AudioSegment.converter = params["ffmpeg_path"]

            args = argparse.Namespace(
                schedule=params.get("schedule"),
                outfile=params.get("outfile", "session.wav"),
                duration=float(params.get("duration", 60)),
                base=float(params.get("base", 0)),
                beat=float(params.get("beat", 0)),
                noise=params.get("noise"),
                isochronic=params.get("isochronic"),
                harmonic_box=params.get("harmonic_box"),
                music=params.get("music"),
                music_amp=float(params.get("music_amp", 100.0))
            )

            audio = sbagen.generate_audio(args)

            if audio is not None:
                # Normalize to prevent clipping before writing
                max_val = np.max(np.abs(audio))
                if max_val > 1.0:
                    audio /= max_val
                sf.write(args.outfile, audio, sbagen.SAMPLE_RATE)
                return f"Wrote {len(audio) / sbagen.SAMPLE_RATE:.2f}s to {args.outfile}"
            else:
                return "No audio generated."

        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

class SbagenGui:
    def __init__(self, root):
        self.controller = SbagenController()
        self.root = root
        root.title("SBAGEN GUI")

        main_label = ttk.Label(root, text="SBAGEN GUI", font=("Helvetica", 16))
        main_label.pack(pady=10)

        notebook = ttk.Notebook(root)
        notebook.pack(padx=10, pady=10, fill="both", expand=True)

        quick_tab = ttk.Frame(notebook)
        schedule_tab = ttk.Frame(notebook)
        advanced_tab = ttk.Frame(notebook)

        notebook.add(quick_tab, text="Quick Generate")
        notebook.add(schedule_tab, text="Schedule File")
        notebook.add(advanced_tab, text="Advanced")

        # --- Quick Generate Tab ---
        # Input fields
        self.base_freq = self.create_input(quick_tab, "Base Frequency (Hz):", "200")
        self.beat_freq = self.create_input(quick_tab, "Beat Frequency (Hz):", "10")
        self.duration = self.create_input(quick_tab, "Duration (s):", "60")
        self.outfile = self.create_input(quick_tab, "Output File:", "session.wav")

        # Generate button
        generate_button = ttk.Button(quick_tab, text="Generate", command=self.run_generate)
        generate_button.pack(pady=5)

        # --- Schedule File Tab ---
        self.schedule_file = tk.StringVar()
        schedule_label = ttk.Label(schedule_tab, textvariable=self.schedule_file)
        schedule_label.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.schedule_file.set("No file selected.")

        browse_button = ttk.Button(schedule_tab, text="Browse...", command=self.browse_file)
        browse_button.pack(side="left", padx=5)

        schedule_generate_button = ttk.Button(schedule_tab, text="Generate from Schedule", command=self.run_generate_from_schedule)
        schedule_generate_button.pack(side="left", padx=5)

        # --- Advanced Tab ---
        # Isochronic tones
        iso_frame = ttk.LabelFrame(advanced_tab, text="Isochronic Tone")
        iso_frame.pack(padx=10, pady=10, fill="x")
        iso_desc = ttk.Label(iso_frame, text="A single tone that is turned on and off rapidly.")
        iso_desc.pack(pady=5)
        self.iso_freq = self.create_input(iso_frame, "Frequency (Hz):", "200")
        self.iso_beat = self.create_input(iso_frame, "Beat (Hz):", "10")

        # Harmonic Box X
        hbox_frame = ttk.LabelFrame(advanced_tab, text="Harmonic Box X")
        hbox_frame.pack(padx=10, pady=10, fill="x")
        hbox_desc = ttk.Label(hbox_frame, text="A more complex tone with harmonic layers.")
        hbox_desc.pack(pady=5)
        self.hbox_base = self.create_input(hbox_frame, "Base Freq:", "180")
        self.hbox_diff = self.create_input(hbox_frame, "Difference:", "5")
        self.hbox_mod = self.create_input(hbox_frame, "Modulator:", "8")

        # Noise
        noise_frame = ttk.LabelFrame(advanced_tab, text="Noise")
        noise_frame.pack(padx=10, pady=10, fill="x")
        self.noise_amp = self.create_input(noise_frame, "White Noise Amp (%):", "0")

        # Background music
        music_frame = ttk.LabelFrame(advanced_tab, text="Background Music")
        music_frame.pack(padx=10, pady=10, fill="x")
        self.music_file = tk.StringVar()
        music_label = ttk.Label(music_frame, textvariable=self.music_file)
        music_label.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.music_file.set("No file selected.")
        music_browse_button = ttk.Button(music_frame, text="Browse...", command=self.browse_music_file)
        music_browse_button.pack(side="left", padx=5)
        self.music_amp = self.create_input(music_frame, "Volume (%):", "50")

        # FFMPEG Path
        ffmpeg_frame = ttk.LabelFrame(advanced_tab, text="FFMPEG Path")
        ffmpeg_frame.pack(padx=10, pady=10, fill="x")
        self.ffmpeg_path = self.create_input(ffmpeg_frame, "FFMPEG Executable:", "")
        ffmpeg_browse_button = ttk.Button(ffmpeg_frame, text="Browse...", command=self.browse_ffmpeg)
        ffmpeg_browse_button.pack(side="left", padx=5)


        advanced_generate_button = ttk.Button(advanced_tab, text="Generate Advanced", command=self.run_generate_advanced)
        advanced_generate_button.pack(pady=10)

        # Status bar
        self.status = tk.StringVar()
        self.status.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    def create_input(self, parent, label_text, default_value):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        label = ttk.Label(frame, text=label_text, width=20)
        label.pack(side="left")
        entry = ttk.Entry(frame)
        entry.pack(side="left", expand=True, fill="x")
        entry.insert(0, default_value)
        return entry

    def run_generate(self):
        self.status.set("Generating audio...")
        params = {
            "base": self.base_freq.get(),
            "beat": self.beat_freq.get(),
            "duration": self.duration.get(),
            "outfile": self.outfile.get()
        }
        result = self.controller.generate(params)
        self.status.set(result)
        if "Error" in result:
            messagebox.showerror("Error", result)
        else:
            messagebox.showinfo("Success", result)

    def run_generate_advanced(self):
        self.status.set("Generating advanced audio...")
        params = {
            "duration": self.duration.get(),
            "outfile": self.outfile.get(),
            "isochronic": (float(self.iso_freq.get()), float(self.iso_beat.get())),
            "harmonic_box": (float(self.hbox_base.get()), float(self.hbox_diff.get()), float(self.hbox_mod.get())),
            "music": self.music_file.get() if self.music_file.get() != "No file selected." else None,
            "music_amp": self.music_amp.get(),
            "ffmpeg_path": self.ffmpeg_path.get(),
            "noise": float(self.noise_amp.get()) if self.noise_amp.get() else None
        }
        result = self.controller.generate(params)
        self.status.set(result)
        if "Error" in result:
            messagebox.showerror("Error", result)
        else:
            messagebox.showinfo("Success", result)

    def browse_file(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Select an SBG file",
            filetypes=(("SBG files", "*.sbg"), ("All files", "*.*"))
        )
        if filepath:
            self.schedule_file.set(filepath)

    def browse_music_file(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Select a music file",
            filetypes=(("Audio files", "*.wav *.ogg *.mp3"), ("All files", "*.*"))
        )
        if filepath:
            self.music_file.set(filepath)

    def browse_ffmpeg(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(title="Select FFMPEG executable")
        if filepath:
            self.ffmpeg_path.delete(0, tk.END)
            self.ffmpeg_path.insert(0, filepath)

    def run_generate_from_schedule(self):
        filepath = self.schedule_file.get()
        if not filepath or filepath == "No file selected.":
            messagebox.showwarning("Warning", "Please select a schedule file first.")
            return

        self.status.set("Generating from schedule...")
        params = {
            "schedule": filepath,
            "outfile": self.outfile.get(),
            # Duration is typically determined by the schedule itself
        }
        result = self.controller.generate(params)
        self.status.set(result)
        if "Error" in result:
            messagebox.showerror("Error", result)
        else:
            messagebox.showinfo("Success", result)


if __name__ == "__main__":
    root = tk.Tk()
    app = SbagenGui(root)
    root.mainloop()
