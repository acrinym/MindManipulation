import tkinter as tk
from tkinter import ttk, messagebox
import argparse
from pysbagen import cli as pysbagen_cli
from pysbagen.mixer import mix_generators
from pysbagen.generators.generic import GenericToneSpec
from pysbagen.parser import parse_sbg_from_string
import visualization as viz
import pyaudio
import threading
import drg_decoder
from PIL import Image, ImageTk
import io

class SbagenController:
    def generate(self, params):
        try:
            # This is a bit of a hack to reuse the CLI parser
            # A better solution would be to have a proper API
            args = []
            if params.get("outfile"):
                args.extend(["-o", params["outfile"]])
            if params.get("duration"):
                args.extend(["-d", str(params["duration"])])
            if params.get("base") and params.get("beat"):
                args.extend(["--base", str(params["base"]), "--beat", str(params["beat"])])
            if params.get("isochronic"):
                args.extend(["--isochronic", str(params["isochronic"][0]), str(params["isochronic"][1])])
            if params.get("harmonic_box"):
                args.extend(["--harmonic-box", str(params["harmonic_box"][0]), str(params["harmonic_box"][1]), str(params["harmonic_box"][2])])
            if params.get("noise"):
                args.extend(["--noise", str(params["noise"])])
            if params.get("music"):
                args.extend(["--music", params["music"]])
            if params.get("music_amp"):
                args.extend(["--music-amp", str(params["music_amp"])])
            if params.get("schedule"):
                args.append(params["schedule"])

            # We need to temporarily replace sys.argv to use the parser
            import sys
            original_argv = sys.argv
            sys.argv = ["sbgpy"] + args
            pysbagen_cli.main()
            sys.argv = original_argv

            return f"Successfully generated {params.get('outfile', 'session.wav')}"

        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def generate_with_viz(self, params):
        # This is also a hack. A proper API would be better.
        gens = []
        if params.get("isochronic"):
            gens.append(GenericToneSpec(freq=params["isochronic"][0], amp=100.0, waveform="sine"))

        yield from mix_generators(gens, params["duration"])


    def generate_tones(self, specs, duration):
        gens = []
        for spec in specs:
            gens.append(GenericToneSpec(
                freq=spec["freq"],
                amp=spec["amp"],
                waveform=spec["waveform"]
            ))
        yield from mix_generators(gens, duration)

class SbagenGui:
    def __init__(self, root):
        self.controller = SbagenController()
        self.root = root
        root.title("SBAGEN GUI")

        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open I-Doser file...", command=self.open_drg_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        main_label = ttk.Label(root, text="SBAGEN GUI", font=("Helvetica", 16))
        main_label.pack(pady=10)

        notebook = ttk.Notebook(root)
        notebook.pack(padx=10, pady=10, fill="both", expand=True)

        quick_tab = ttk.Frame(notebook)
        schedule_tab = ttk.Frame(notebook)
        advanced_tab = ttk.Frame(notebook)
        self.viz_tab = ttk.Frame(notebook)
        tone_gen_tab = ttk.Frame(notebook)

        notebook.add(quick_tab, text="Quick Generate")
        notebook.add(schedule_tab, text="Schedule File")
        notebook.add(advanced_tab, text="Advanced")
        notebook.add(self.viz_tab, text="Visualization")
        notebook.add(tone_gen_tab, text="Tone Generator")

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
        self.image_label = ttk.Label(schedule_tab)
        self.image_label.pack(pady=5)
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

        # --- Tone Generator Tab ---
        self.tone_generators = []
        self.tone_canvas = tk.Canvas(tone_gen_tab)
        self.tone_frame = ttk.Frame(self.tone_canvas)
        self.tone_scrollbar = ttk.Scrollbar(tone_gen_tab, orient="vertical", command=self.tone_canvas.yview)
        self.tone_canvas.configure(yscrollcommand=self.tone_scrollbar.set)

        self.tone_scrollbar.pack(side="right", fill="y")
        self.tone_canvas.pack(side="left", fill="both", expand=True)
        self.tone_canvas.create_window((0,0), window=self.tone_frame, anchor="nw")

        self.tone_frame.bind("<Configure>", lambda e: self.tone_canvas.configure(scrollregion=self.tone_canvas.bbox("all")))

        soundscape_frame = ttk.LabelFrame(tone_gen_tab, text="Background Soundscape")
        soundscape_frame.pack(padx=10, pady=10, fill="x")
        self.soundscape_file = tk.StringVar()
        soundscape_label = ttk.Label(soundscape_frame, textvariable=self.soundscape_file)
        soundscape_label.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.soundscape_file.set("No file selected.")
        load_soundscape_button = ttk.Button(soundscape_frame, text="Load...", command=self.load_soundscape)
        load_soundscape_button.pack(side="left", padx=5)
        self.soundscape_amp = tk.Scale(soundscape_frame, from_=0, to=100, orient="horizontal", label="Volume (%)")
        self.soundscape_amp.set(50)
        self.soundscape_amp.pack(side="left", padx=5)

        add_tone_button = ttk.Button(tone_gen_tab, text="Add Tone", command=self.add_tone_generator)
        add_tone_button.pack(pady=5)

        generate_tones_button = ttk.Button(tone_gen_tab, text="Generate Tones", command=self.run_generate_tones)
        generate_tones_button.pack(pady=5)

        # --- Visualization Tab ---
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        viz_button = ttk.Button(self.viz_tab, text="Play with Visualization", command=self.run_generate_and_viz)
        viz_button.pack(pady=10)

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

    def run_generate_and_viz(self):
        self.status.set("Starting visualization thread...")
        thread = threading.Thread(target=self._generate_and_viz_thread)
        thread.daemon = True
        thread.start()

    def _generate_and_viz_thread(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=2,
                        rate=44100, # Use constant
                        output=True)

        params = {
            "duration": float(self.duration.get()),
            "outfile": self.outfile.get(),
            "isochronic": (float(self.iso_freq.get()), float(self.iso_beat.get())),
            "harmonic_box": (float(self.hbox_base.get()), float(self.hbox_diff.get()), float(self.hbox_mod.get())),
            "music": self.music_file.get() if self.music_file.get() != "No file selected." else None,
            "music_amp": self.music_amp.get(),
            "ffmpeg_path": self.ffmpeg_path.get(),
            "noise": float(self.noise_amp.get()) if self.noise_amp.get() else None
        }

        try:
            for chunk, info in self.controller.generate_with_viz(params):
                stream.write(chunk.astype(np.float32).tobytes())

                # Map the frequency to (n, m) parameters
                if info and info[0]['type'] == 'isochronic':
                    freq = info[0]['freq']
                    n, m = viz.map_freq_to_params(freq)

                    self.fig.clear()
                    new_fig = viz.generate_chladni_pattern(n, m)
                    self.canvas.figure = new_fig
                    self.canvas.draw()

            self.status.set("Visualization complete.")
        except Exception as e:
            self.status.set(f"Error: {e}")
            # Since this is in a thread, messagebox will not work well.
            # We will just print the error for now.
            print(f"Error in visualization thread: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

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

    def open_drg_file(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Select an I-Doser file",
            filetypes=(("I-Doser files", "*.drg"), ("All files", "*.*"))
        )
        if not filepath:
            return

        try:
            sbg_string, image_data = drg_decoder.decode_drg(filepath)
            tones, sched = parse_sbg_from_string(sbg_string)

            if image_data:
                image = Image.open(io.BytesIO(image_data))
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo # Keep a reference
            else:
                self.image_label.config(image="")
                self.image_label.image = None

            # For now, just print the parsed data
            print("Successfully parsed .drg file:")
            print("Tones:", tones)
            print("Schedule:", sched)
            self.status.set(f"Successfully parsed {filepath}")
        except Exception as e:
            self.status.set(f"Error parsing .drg file: {e}")
            messagebox.showerror("Error", f"Could not parse .drg file: {e}")

    def add_tone_generator(self):
        frame = ttk.Frame(self.tone_frame, padding=5, relief="groove", borderwidth=2)
        frame.pack(fill="x", pady=5)

        tone_num = len(self.tone_generators) + 1
        label = ttk.Label(frame, text=f"Tone {tone_num}")
        label.pack()

        freq_slider = tk.Scale(frame, from_=20, to=1000, orient="horizontal", label="Frequency (Hz)")
        freq_slider.set(200)
        freq_slider.pack(fill="x")

        amp_slider = tk.Scale(frame, from_=0, to=100, orient="horizontal", label="Amplitude (%)")
        amp_slider.set(50)
        amp_slider.pack(fill="x")

        waveform = tk.StringVar(frame)
        waveform.set("sine")
        waveform_menu = ttk.OptionMenu(frame, waveform, "sine", "square", "triangle", "sawtooth")
        waveform_menu.pack()

        remove_button = ttk.Button(frame, text="Remove", command=lambda: self.remove_tone_generator(frame))
        remove_button.pack()

        self.tone_generators.append({
            "frame": frame,
            "freq": freq_slider,
            "amp": amp_slider,
            "waveform": waveform
        })

    def run_generate_tones(self):
        self.status.set("Generating tones...")

        specs = []
        for gen in self.tone_generators:
            spec = {
                "freq": gen["freq"].get(),
                "amp": gen["amp"].get(),
                "waveform": gen["waveform"].get()
            }
            specs.append(spec)

        if not specs:
            self.status.set("No tones to generate.")
            return

        thread = threading.Thread(target=self._generate_tones_thread, args=(specs,))
        thread.daemon = True
        thread.start()

    def _generate_tones_thread(self, specs):
        # This is a simplified version of the viz thread for now
        # It just plays the audio without visualization
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=2,
                        rate=44100, # Use constant
                        output=True)

        try:
            # Duration is fixed for now
            duration = 10
            generator = self.controller.generate_tones(specs, duration)

            soundscape_path = self.soundscape_file.get()
            if soundscape_path and soundscape_path != "No file selected.":
                soundscape_amp = self.soundscape_amp.get()
                soundscape_spec = pysbagen.generators.file.FileSpec(soundscape_path, soundscape_amp)
                soundscape_gen = soundscape_spec.generator(duration, loop=True)

                for chunk, info in generator:
                    soundscape_chunk, _ = next(soundscape_gen)
                    mixed_chunk = chunk + soundscape_chunk
                    stream.write(mixed_chunk.astype(np.float32).tobytes())
            else:
                for chunk, info in generator:
                    stream.write(chunk.astype(np.float32).tobytes())

            self.status.set("Tone generation complete.")
        except Exception as e:
            self.status.set(f"Error: {e}")
            print(f"Error in tone generation thread: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def remove_tone_generator(self, frame):
        for i, gen in enumerate(self.tone_generators):
            if gen["frame"] == frame:
                gen["frame"].destroy()
                self.tone_generators.pop(i)
                break

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
