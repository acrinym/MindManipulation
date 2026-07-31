from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from pysbagen.api import render_schedule, write_audio
from pysbagen.compatibility import RenderDisposition
from pysbagen.importers import ImportedArtifact, import_artifact
from pysbagen.inspector import build_timeline, inspect_audio_source, timeline_to_text
from pysbagen.library import LocalLibrary


class CompatibilityInspectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PySbagen Compatibility Inspector")
        self.root.geometry("1050x760")
        self.artifact: ImportedArtifact | None = None
        self.source_var = tk.StringVar()
        self.duration_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose an SBG or DRG file to inspect.")
        self.ack_var = tk.BooleanVar(value=False)
        self._busy = False
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Artifact").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse…", command=self._browse).grid(row=0, column=2)
        ttk.Button(top, text="Inspect", command=self.inspect).grid(row=0, column=3, padx=(6, 0))
        top.columnconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Render duration (seconds; required for open schedules)").pack(side="left")
        ttk.Entry(controls, textvariable=self.duration_var, width=12).pack(side="left", padx=6)
        self.ack = ttk.Checkbutton(
            controls,
            text="I reviewed and accept disclosed compatibility changes",
            variable=self.ack_var,
            command=self._update_actions,
        )
        self.ack.pack(side="left", padx=12)
        self.render_button = ttk.Button(controls, text="Render inspected artifact…", command=self.render)
        self.render_button.pack(side="right")
        self.library_button = ttk.Button(controls, text="Add to local library", command=self.add_to_library)
        self.library_button.pack(side="right", padx=6)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        report_frame = ttk.Frame(notebook)
        timeline_frame = ttk.Frame(notebook)
        package_frame = ttk.Frame(notebook)
        source_frame = ttk.Frame(notebook)
        notebook.add(report_frame, text="Import report")
        notebook.add(timeline_frame, text="Timeline")
        notebook.add(package_frame, text="Package elements")
        notebook.add(source_frame, text="Audio sources")

        self.report_text = tk.Text(report_frame, wrap="word")
        self.report_text.pack(fill="both", expand=True)
        self.timeline_text = tk.Text(timeline_frame, wrap="none")
        self.timeline_text.pack(fill="both", expand=True)

        self.package_tree = ttk.Treeview(package_frame, columns=("role", "size", "hash", "media"), show="headings")
        for column, title, width in (("role", "Role", 180), ("size", "Bytes", 90), ("hash", "SHA-256", 360), ("media", "Media type", 150)):
            self.package_tree.heading(column, text=title)
            self.package_tree.column(column, width=width, anchor="w")
        self.package_tree.pack(fill="both", expand=True)

        self.source_tree = ttk.Treeview(source_frame, columns=("state", "channels", "rate", "duration", "path"), show="headings")
        for column, title, width in (("state", "State", 120), ("channels", "Channels", 80), ("rate", "Rate", 90), ("duration", "Duration", 100), ("path", "Path", 540)):
            self.source_tree.heading(column, text=title)
            self.source_tree.column(column, width=width, anchor="w")
        self.source_tree.pack(fill="both", expand=True)

        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x")
        self._update_actions()

    def _browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose an SBG or DRG artifact",
            filetypes=[("SBaGen and I-Doser", "*.sbg *.drg"), ("All files", "*.*")],
        )
        if selected:
            self.source_var.set(selected)
            self.inspect()

    def inspect(self) -> None:
        if self._busy:
            return
        source = self.source_var.get().strip()
        if not source:
            messagebox.showerror("Missing artifact", "Choose an SBG or DRG file first.")
            return
        try:
            artifact = import_artifact(source)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        self.artifact = artifact
        self.ack_var.set(False)
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", artifact.report.to_text())
        self.timeline_text.delete("1.0", "end")
        self.timeline_text.insert("1.0", timeline_to_text(build_timeline(artifact)))
        self._fill_packages(artifact)
        self._fill_sources(artifact)
        self.status_var.set(
            f"Inspected {Path(source).name}: {artifact.report.render_disposition.value}. "
            "Rendering remains locked unless this report permits it."
        )
        self._update_actions()

    def _fill_packages(self, artifact: ImportedArtifact) -> None:
        self.package_tree.delete(*self.package_tree.get_children())
        for element in artifact.report.package_elements:
            self.package_tree.insert("", "end", values=(element.role, element.size, element.sha256, element.media_type or ""))

    def _fill_sources(self, artifact: ImportedArtifact) -> None:
        self.source_tree.delete(*self.source_tree.get_children())
        seen: set[str] = set()
        for specs in artifact.tone_sets.values():
            for spec in specs:
                path = getattr(spec, "path", None)
                if not path or path in seen:
                    continue
                seen.add(path)
                report = inspect_audio_source(path)
                self.source_tree.insert(
                    "",
                    "end",
                    values=(
                        report.state.value,
                        report.channels if report.channels is not None else "?",
                        report.sample_rate if report.sample_rate is not None else "?",
                        f"{report.duration:.2f}s" if report.duration is not None else "?",
                        report.path,
                    ),
                )

    def _update_actions(self) -> None:
        artifact = self.artifact
        render_enabled = False
        if artifact is not None and not self._busy:
            disposition = artifact.report.render_disposition
            render_enabled = disposition is RenderDisposition.SAFE or (
                disposition is RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES and self.ack_var.get()
            )
        self.render_button.configure(state="normal" if render_enabled else "disabled")
        self.library_button.configure(state="normal" if artifact is not None and not self._busy else "disabled")
        self.ack.configure(
            state=(
                "normal"
                if artifact is not None
                and artifact.report.render_disposition is RenderDisposition.SAFE_WITH_DISCLOSED_CHANGES
                else "disabled"
            )
        )

    def render(self) -> None:
        artifact = self.artifact
        if artifact is None:
            return
        destination = filedialog.asksaveasfilename(
            title="Render inspected artifact",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not destination:
            return
        try:
            duration = float(self.duration_var.get()) if self.duration_var.get().strip() else None
            artifact.report.require_renderable(allow_disclosed_changes=self.ack_var.get())
            artifact.require_duration(duration)
        except ValueError as exc:
            messagebox.showerror("Render blocked", str(exc))
            return
        self._set_busy(True, "Rendering inspected artifact…")

        def worker() -> None:
            try:
                chunks = render_schedule(
                    self.source_var.get(),
                    duration,
                    allow_disclosed_changes=self.ack_var.get(),
                )
                result = write_audio(chunks, destination)
            except Exception as exc:
                self.root.after(0, lambda: self._finish_error("Render failed", exc))
                return
            self.root.after(0, lambda: self._finish_success(f"Rendered {result.duration:.2f}s to {result.outfile}"))

        threading.Thread(target=worker, daemon=True).start()

    def add_to_library(self) -> None:
        if self.artifact is None:
            return
        try:
            item = LocalLibrary().add(self.artifact)
        except Exception as exc:
            messagebox.showerror("Library import failed", str(exc))
            return
        self.status_var.set(f"Stored local library item {item.item_id} as {item.state}.")

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self.status_var.set(status)
        self._update_actions()

    def _finish_error(self, title: str, error: Exception) -> None:
        self._set_busy(False, str(error))
        messagebox.showerror(title, str(error))

    def _finish_success(self, status: str) -> None:
        self._set_busy(False, status)
        messagebox.showinfo("Render complete", status)


def main() -> int:
    root = tk.Tk()
    CompatibilityInspectorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
