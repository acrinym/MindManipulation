from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compatibility import sha256_bytes
from .drg import preserve_drg_package
from .importers import ImportedArtifact
from .inspector import build_timeline, timeline_to_dict

LIBRARY_STATES = {"available", "imported", "missing-source", "superseded", "archived", "incompatible", "withdrawn", "research-only"}


@dataclass(frozen=True)
class LibraryItem:
    item_id: str
    path: Path
    manifest: dict[str, Any]

    @property
    def state(self) -> str:
        return str(self.manifest.get("state", "incompatible"))


class LocalLibrary:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else default_library_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, artifact: ImportedArtifact, *, state: str | None = None, tags: list[str] | None = None, notes: str | None = None) -> LibraryItem:
        resolved_state = state or _state_from_artifact(artifact)
        if resolved_state not in LIBRARY_STATES:
            raise ValueError(f"Unknown library state: {resolved_state}")
        item_id = artifact.report.source_sha256
        destination = self.root / item_id
        if destination.exists():
            item = self.get(item_id)
            manifest = dict(item.manifest)
            provenance = dict(manifest.get("provenance") or {})
            records = list(provenance.get("records") or [])
            record = _provenance_record(artifact)
            if not any(existing.get("original_path") == record["original_path"] and existing.get("imported_source_type") == record["imported_source_type"] for existing in records):
                records.append(record)
                provenance["records"] = records
                manifest["provenance"] = provenance
                _write_json_atomic(item.path / "manifest.json", manifest)
            return self.get(item_id)

        temporary = Path(tempfile.mkdtemp(prefix=f".{item_id[:12]}-", dir=self.root))
        try:
            source_path = Path(artifact.report.source_path.split("::", 1)[0])
            source_name = source_path.name or f"source.{artifact.report.source_type}"
            source_dir = temporary / "source"
            source_dir.mkdir()
            source_destination = source_dir / source_name
            if source_path.is_file():
                shutil.copy2(source_path, source_destination)
            elif artifact.source_text is not None:
                source_destination.write_text(artifact.source_text, encoding="utf-8")
            else:
                raise ValueError("Imported artifact has neither a readable source nor preserved source text")
            artifact.report.write_json(temporary / "import-report.json")
            (temporary / "timeline.json").write_text(json.dumps(timeline_to_dict(build_timeline(artifact)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if artifact.package is not None:
                preserve_drg_package(artifact.package, temporary / "package")
            manifest = {
                "schema": "pysbagen.local-library.v1", "item_id": item_id, "state": resolved_state,
                "display_name": source_path.stem or item_id[:12], "source_type": artifact.report.source_type,
                "source": {"original_path": artifact.report.source_path, "stored_name": f"source/{source_name}", "size": artifact.report.source_size, "sha256": artifact.report.source_sha256, "immutable": True},
                "compatibility": {"render_disposition": artifact.report.render_disposition.value, "finding_states": sorted({finding.state.value for finding in artifact.report.findings}), "missing_source_count": len(artifact.report.missing_sources)},
                "recipe": {"editable": bool(artifact.tone_sets and artifact.schedule), "tone_set_count": len(artifact.tone_sets), "event_count": len(artifact.schedule), "inferred_duration": artifact.report.inferred_duration},
                "provenance": {"imported_at": datetime.now(timezone.utc).isoformat(), "importer": "PySbagen compatibility train", "derived_files": ["import-report.json", "timeline.json"], "package_preserved": artifact.package is not None, "records": [_provenance_record(artifact)]},
                "tags": sorted(set(tags or [])), "notes": notes, "supersedes": None, "superseded_by": None,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.get(item_id)

    def get(self, item_id: str) -> LibraryItem:
        path = self.root / item_id
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise KeyError(f"Library item not found: {item_id}")
        return LibraryItem(item_id, path, json.loads(manifest_path.read_text(encoding="utf-8")))

    def list_items(self, *, include_archived: bool = True) -> list[LibraryItem]:
        items = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                item = self.get(path.name)
            except (KeyError, json.JSONDecodeError, OSError):
                continue
            if include_archived or item.state != "archived":
                items.append(item)
        return items

    def set_state(self, item_id: str, state: str) -> LibraryItem:
        if state not in LIBRARY_STATES:
            raise ValueError(f"Unknown library state: {state}")
        item = self.get(item_id)
        manifest = dict(item.manifest)
        manifest["state"] = state
        manifest["state_changed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(item.path / "manifest.json", manifest)
        return self.get(item_id)

    def mark_superseded(self, old_item_id: str, new_item_id: str) -> tuple[LibraryItem, LibraryItem]:
        old_item, new_item = self.get(old_item_id), self.get(new_item_id)
        old_manifest, new_manifest = dict(old_item.manifest), dict(new_item.manifest)
        old_manifest["state"], old_manifest["superseded_by"] = "superseded", new_item_id
        new_manifest["supersedes"] = old_item_id
        changed_at = datetime.now(timezone.utc).isoformat()
        old_manifest["state_changed_at"] = new_manifest["state_changed_at"] = changed_at
        _write_json_atomic(old_item.path / "manifest.json", old_manifest)
        _write_json_atomic(new_item.path / "manifest.json", new_manifest)
        return self.get(old_item_id), self.get(new_item_id)

    def export_manifest(self, item_id: str, destination: str | Path) -> Path:
        item = self.get(item_id)
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "pysbagen.library-export.v1", "exported_at": datetime.now(timezone.utc).isoformat(), "library_manifest": item.manifest, "verification": self.verify(item_id), "import_report": json.loads((item.path / "import-report.json").read_text(encoding="utf-8")), "timeline": json.loads((item.path / "timeline.json").read_text(encoding="utf-8"))}
        _write_json_atomic(target, payload)
        return target

    def verify(self, item_id: str) -> dict[str, Any]:
        item = self.get(item_id)
        source = item.manifest.get("source", {})
        source_path = item.path / str(source.get("stored_name"))
        expected = source.get("sha256")
        actual = sha256_bytes(source_path.read_bytes()) if source_path.is_file() else None
        result = {"item_id": item_id, "source_present": source_path.is_file(), "source_hash_matches": actual == expected, "expected_sha256": expected, "actual_sha256": actual, "report_present": (item.path / "import-report.json").is_file(), "timeline_present": (item.path / "timeline.json").is_file(), "package_manifest_present": (item.path / "package" / "manifest.json").is_file()}
        result["valid"] = all([result["source_present"], result["source_hash_matches"], result["report_present"], result["timeline_present"]])
        return result


def default_library_root() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "PySbagen" / "library"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "pysbagen" / "library"


def _provenance_record(artifact: ImportedArtifact) -> dict[str, Any]:
    return {"original_path": artifact.report.source_path, "imported_at": datetime.now(timezone.utc).isoformat(), "imported_source_type": artifact.report.source_type, "source_size": artifact.report.source_size, "source_sha256": artifact.report.source_sha256}


def _state_from_artifact(artifact: ImportedArtifact) -> str:
    if artifact.report.missing_sources:
        return "missing-source"
    return "incompatible" if artifact.report.render_disposition.value in {"blocked", "inspection-only"} else "imported"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
