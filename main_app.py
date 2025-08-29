import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

from audio_engine import (
    SAMPLE_RATE,
    AnySpec,
    ToneSpec,
    NoiseSpec,
    IsochronicSpec,
    HarmonicBoxSpec,
    build_session,
    parse_sbg,
    mix_generators,
)

class SbaGenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SbaGen-Py GUI")
        self.root.geometry("800x600")

        self.tone_sets = {}
        self.schedule = []
        self.playback_thread = None
        self.stream = None
        self.audio_data = np.array([])
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.create_menu()
        self.create_main_layout(main_frame)
        self.create_playback_controls(main_frame)
        self.create_status_bar(main_frame)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Schedule (.sbg)...", command=self.on_open)
        self.file_menu.add_command(label="Save Session As (.wav)...", command=self.on_save, state=tk.DISABLED)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_exit)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.on_about)

    def create_main_layout(self, parent):
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        session_container = ttk.LabelFrame(top_frame, text="Session Schedule", padding="10")
        session_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.schedule_tree = ttk.Treeview(session_container, columns=("Time", "Action"), show="headings")
        self.schedule_tree.heading("Time", text="Time"); self.schedule_tree.heading("Action", text="Action")
        self.schedule_tree.column("Time", width=100, anchor=tk.W); self.schedule_tree.column("Action", anchor=tk.W)
        vsb = ttk.Scrollbar(session_container, orient="vertical", command=self.schedule_tree.yview)
        self.schedule_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.schedule_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.create_quick_generate_ui(top_frame)

    def create_quick_generate_ui(self, parent):
        qg_container = ttk.LabelFrame(parent, text="Quick Generate", padding="10")
        qg_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.qg_notebook = ttk.Notebook(qg_container)
        self.qg_notebook.pack(fill="both", expand=True, pady=5)
        self.qg_vars = {}

        self.create_qg_tab("Binaural", [("Base Freq (Hz)", "200"), ("Beat Freq (Hz)", "10"), ("Amplitude (%)", "50")])
        self.create_qg_tab("Isochronic", [("Tone Freq (Hz)", "200"), ("Beat Freq (Hz)", "10"), ("Amplitude (%)", "100")])
        self.create_qg_tab("Harmonic Box X", [("Base Freq (Hz)", "180"), ("Difference (Hz)", "5"), ("Modulation (Hz)", "8"), ("Amplitude (%)", "100")])

        common_controls_frame = ttk.Frame(qg_container)
        common_controls_frame.pack(fill=tk.X, pady=5)
        ttk.Label(common_controls_frame, text="Duration (s):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.qg_vars['duration'] = tk.StringVar(value="30")
        ttk.Entry(common_controls_frame, textvariable=self.qg_vars['duration'], width=10).grid(row=0, column=1, sticky=tk.W)

        self.qg_vars['noise_on'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(common_controls_frame, text="Add White Noise", variable=self.qg_vars['noise_on'], command=self.toggle_noise_amp).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5,0))

        self.qg_vars['noise_amp'] = tk.DoubleVar(value=10.0)
        self.qg_noise_slider = ttk.Scale(common_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.qg_vars['noise_amp'], state=tk.DISABLED)
        self.qg_noise_slider.grid(row=2, column=0, columnspan=2, sticky=tk.EW)

        ttk.Button(qg_container, text="▶ Play Quick Session", command=self.play_quick_session).pack(pady=(10,0))

    def create_qg_tab(self, name, fields):
        frame = ttk.Frame(self.qg_notebook, padding=10)
        self.qg_notebook.add(frame, text=name)
        for i, (label, default_val) in enumerate(fields):
            ttk.Label(frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, pady=2)
            var_name = f"{name.lower().replace(' ', '_')}_{label.split(' ')[0].lower()}"
            self.qg_vars[var_name] = tk.StringVar(value=default_val)
            ttk.Entry(frame, textvariable=self.qg_vars[var_name]).grid(row=i, column=1, sticky=tk.EW, padx=5)
            frame.columnconfigure(1, weight=1)

    def toggle_noise_amp(self):
        self.qg_noise_slider.config(state=tk.NORMAL if self.qg_vars['noise_on'].get() else tk.DISABLED)

    def create_playback_controls(self, parent):
        controls_container = ttk.LabelFrame(parent, text="Playback Controls", padding="10")
        controls_container.pack(fill=tk.X, pady=5)
        controls_container.columnconfigure(1, weight=1)
        self.play_button = ttk.Button(controls_container, text="▶ Play File", command=self.play_sbg_session, state=tk.DISABLED)
        self.play_button.grid(row=0, column=0, padx=(0, 5))
        self.pause_button = ttk.Button(controls_container, text="❚❚ Pause", command=self.pause_session, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=2, padx=5)
        self.stop_button = ttk.Button(controls_container, text="■ Stop", command=self.stop_session, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=5)
        self.progress_var = tk.DoubleVar()
        self.progressbar = ttk.Progressbar(controls_container, variable=self.progress_var, maximum=100)
        self.progressbar.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(controls_container, text="Volume:").grid(row=0, column=4, padx=(10, 5))
        self.volume_var = tk.DoubleVar(value=75.0)
        self.volume_slider = ttk.Scale(controls_container, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.volume_var)
        self.volume_slider.grid(row=0, column=5, sticky="ew")

    def create_status_bar(self, parent):
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        self.status_var = tk.StringVar(value="Ready. Open an .sbg file or use Quick Generate.")
        ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, padx=5, pady=2)

    def display_schedule(self):
        for item in self.schedule_tree.get_children(): self.schedule_tree.delete(item)
        for time_val, actions in self.schedule:
            m, s = divmod(time_val, 60); h, m = divmod(m, 60)
            self.schedule_tree.insert("", tk.END, values=(f"{h:02d}:{m:02d}:{s:02d}", " ".join(actions)))

    def on_open(self):
        if self.is_playing: self.stop_session()
        filepath = filedialog.askopenfilename(title="Open SBG File", filetypes=(("SBG Schedules", "*.sbg"), ("All files", "*.*")))
        if not filepath: return
        try:
            self.status_var.set(f"Loading {filepath}...")
            self.tone_sets, self.schedule = parse_sbg(filepath)
            self.display_schedule()
            self.play_button.config(state=tk.NORMAL)
            self.file_menu.entryconfig("Save Session As (.wav)...", state=tk.NORMAL)
            self.status_var.set(f"Loaded successfully: {filepath}")
        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to parse the file: {e}")
            self.status_var.set("Error loading file.")

    def audio_callback(self, outdata, frames, time, status):
        if status: print(f"Stream status: {status}")
        if self.is_paused: outdata.fill(0); return
        remaining = len(self.audio_data) - self.current_frame
        if remaining >= frames:
            chunk = self.audio_data[self.current_frame : self.current_frame + frames]
            outdata[:] = chunk * (self.volume_var.get() / 100.0)
            self.current_frame += frames
            progress = (self.current_frame / len(self.audio_data)) * 100
            self.root.after(0, self.progress_var.set, progress)
        else:
            outdata[:remaining] = self.audio_data[self.current_frame:] * (self.volume_var.get() / 100.0)
            outdata[remaining:].fill(0)
            self.root.after(0, self.stop_session); raise sd.CallbackStop

    def start_playback_thread(self, audio_data):
        if self.is_playing: messagebox.showwarning("Busy", "Another session is already playing."); return
        self.audio_data = audio_data; self.current_frame = 0
        self.play_button.config(state=tk.DISABLED); self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL); self.is_playing = True; self.is_paused = False
        self.status_var.set("Starting playback...")
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _playback_worker(self):
        try:
            with sd.OutputStream(samplerate=SAMPLE_RATE, channels=self.audio_data.shape[1], callback=self.audio_callback, dtype='float32') as self.stream:
                while self.is_playing and self.stream.active: time.sleep(0.1)
        except Exception as e:
            print(f"Error during playback: {e}"); self.root.after(0, messagebox.showerror, "Playback Error", str(e))
            self.root.after(0, self.stop_session)

    def play_sbg_session(self):
        if not self.schedule: messagebox.showinfo("No Schedule", "Please open an .sbg file first."); return
        self.status_var.set("Generating audio from schedule...")
        audio = build_session(self.tone_sets, self.schedule.copy(), duration=None)
        self.start_playback_thread(audio)

    def _generate_quick_session_audio(self):
        try:
            duration = float(self.qg_vars['duration'].get())
            if duration <= 0: raise ValueError("Duration must be positive.")
            selected_tab = self.qg_notebook.tab(self.qg_notebook.select(), "text")
            gens: list[AnySpec] = []
            if selected_tab == "Binaural":
                gens.append(ToneSpec(float(self.qg_vars['binaural_base'].get()), float(self.qg_vars['binaural_beat'].get()), float(self.qg_vars['binaural_amplitude'].get())))
            elif selected_tab == "Isochronic":
                gens.append(IsochronicSpec(float(self.qg_vars['isochronic_tone'].get()), float(self.qg_vars['isochronic_beat'].get()), float(self.qg_vars['isochronic_amplitude'].get())))
            elif selected_tab == "Harmonic Box X":
                gens.append(HarmonicBoxSpec(float(self.qg_vars['harmonic_box_x_base'].get()), float(self.qg_vars['harmonic_box_x_difference'].get()), float(self.qg_vars['harmonic_box_x_modulation'].get()), float(self.qg_vars['harmonic_box_x_amplitude'].get())))
            if self.qg_vars['noise_on'].get(): gens.append(NoiseSpec(self.qg_vars['noise_amp'].get()))
            return mix_generators(gens, duration)
        except (ValueError, KeyError) as e:
            messagebox.showerror("Invalid Input", f"Please check your input values. Error: {e}")
            return None

    def play_quick_session(self):
        self.status_var.set("Generating quick session audio...")
        audio = self._generate_quick_session_audio()
        if audio is not None: self.start_playback_thread(audio)

    def on_save(self):
        self.status_var.set("Preparing to save...")
        audio_to_save = None
        if self.schedule:
            self.status_var.set("Generating audio from loaded schedule...")
            audio_to_save = build_session(self.tone_sets, self.schedule.copy(), duration=None)
        else:
            self.status_var.set("Generating audio from Quick Generate settings...")
            audio_to_save = self._generate_quick_session_audio()

        if audio_to_save is None:
            self.status_var.set("Audio generation failed. Nothing to save."); return

        filepath = filedialog.asksaveasfilename(title="Save WAV File", defaultextension=".wav", filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))
        if not filepath: self.status_var.set("Save cancelled."); return

        try:
            self.status_var.set(f"Saving to {filepath}...")
            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_to_save))
            if max_val > 1.0: audio_to_save /= max_val
            sf.write(filepath, audio_to_save, SAMPLE_RATE)
            self.status_var.set(f"Successfully saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file: {e}")
            self.status_var.set("Error saving file.")

    def pause_session(self):
        if not self.is_playing: return
        self.is_paused = not self.is_paused
        self.pause_button.config(text="▶ Resume" if self.is_paused else "❚❚ Pause")
        self.status_var.set("Playback Paused" if self.is_paused else "Playing...")

    def stop_session(self):
        self.is_playing = False
        if self.stream is not None: self.stream.stop(); self.stream.close(); self.stream = None
        self.audio_data = np.array([]); self.current_frame = 0
        self.status_var.set("Playback Stopped. Ready.")
        self.progress_var.set(0)
        self.play_button.config(state=tk.NORMAL if self.schedule else tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED, text="❚❚ Pause")
        self.stop_button.config(state=tk.DISABLED)

    def on_about(self): messagebox.showinfo("About SbaGen-Py GUI", "SbaGen-Py GUI\n\nA modern interface for the SBAGEN audio generator.")

    def on_exit(self):
        if self.is_playing and messagebox.askyesno("Exit", "A session is playing. Are you sure you want to exit?"):
            self.stop_session(); self.root.quit()
        elif not self.is_playing: self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = SbaGenApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()
