from __future__ import annotations

import base64
import binascii
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drg_decoder import rc4_decrypt

from .compatibility import PackageElement, sha256_bytes


@dataclass(frozen=True)
class DrgElement:
    index: int
    role: str
    encoded_text: str
    decoded_bytes: bytes
    encoding: str
    decrypted_bytes: bytes | None = None
    media_type: str | None = None

    @property
    def effective_bytes(self) -> bytes:
        return self.decrypted_bytes if self.decrypted_bytes is not None else self.decoded_bytes

    def to_package_element(self, stored_name: str | None = None) -> PackageElement:
        data = self.effective_bytes
        preview = _text_preview(data)
        return PackageElement(
            index=self.index,
            role=self.role,
            size=len(data),
            sha256=sha256_bytes(data),
            encoding=self.encoding,
            media_type=self.media_type,
            text_preview=preview,
            stored_name=stored_name,
        )


@dataclass
class DrgPackage:
    source_path: Path
    source_bytes: bytes
    text_encoding: str
    header: str
    elements: list[DrgElement]
    schedule_text: str | None
    image_bytes: bytes | None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def source_sha256(self) -> str:
        return sha256_bytes(self.source_bytes)

    def package_elements(self) -> list[PackageElement]:
        return [element.to_package_element() for element in self.elements]


def parse_drg_package(path: str | Path) -> DrgPackage:
    source_path = Path(path).expanduser().resolve()
    data = source_path.read_bytes()
    text, encoding = _decode_container_text(data)
    return parse_drg_bytes(data, source_path=source_path, container_text=text, encoding=encoding)


def parse_drg_bytes(
    data: bytes,
    *,
    source_path: str | Path = "memory.drg",
    container_text: str | None = None,
    encoding: str | None = None,
) -> DrgPackage:
    path = Path(source_path)
    if container_text is None or encoding is None:
        container_text, encoding = _decode_container_text(data)

    first_line, separator, remainder = container_text.partition("\n")
    header = first_line.rstrip("\r")
    payload_text = remainder if separator else ""
    raw_parts = payload_text.split("@") if payload_text else []
    while raw_parts and not raw_parts[0].strip():
        raw_parts.pop(0)

    warnings: list[str] = []
    elements: list[DrgElement] = [
        DrgElement(
            index=0,
            role="header",
            encoded_text=header,
            decoded_bytes=header.encode(encoding, errors="replace"),
            encoding=encoding,
            media_type="text/plain",
        )
    ]

    for payload_index, raw_part in enumerate(raw_parts, start=1):
        compact = "".join(raw_part.split())
        if not compact:
            decoded = b""
            element_encoding = "empty"
        else:
            try:
                decoded = base64.b64decode(compact, validate=True)
                element_encoding = "base64"
            except (binascii.Error, ValueError):
                decoded = raw_part.encode(encoding, errors="replace")
                element_encoding = "raw-text"
                warnings.append(f"Element {payload_index} was not valid base64 and was preserved verbatim.")

        role = _legacy_role(payload_index)
        decrypted: bytes | None = None
        media_type: str | None = None
        if role in {"image", "schedule"} and decoded:
            decrypted = rc4_decrypt(decoded)
            if role == "image":
                decrypted = _decode_nested_base64(decrypted)
                media_type = _sniff_media_type(decrypted)
            else:
                media_type = "text/plain"
        else:
            media_type = _sniff_media_type(decoded)

        elements.append(
            DrgElement(
                index=payload_index,
                role=role,
                encoded_text=raw_part,
                decoded_bytes=decoded,
                encoding=element_encoding,
                decrypted_bytes=decrypted,
                media_type=media_type,
            )
        )

    schedule_text = None
    image_bytes = None
    metadata: dict[str, Any] = {"header": header}
    for element in elements:
        if element.role == "schedule" and element.decrypted_bytes is not None:
            schedule_text, schedule_encoding = _decode_schedule(element.decrypted_bytes)
            metadata["schedule_encoding"] = schedule_encoding
        elif element.role == "image" and element.decrypted_bytes is not None:
            image_bytes = element.decrypted_bytes
        elif element.role.startswith("metadata"):
            parsed = _parse_metadata(element.effective_bytes)
            if parsed:
                metadata[element.role] = parsed

    if schedule_text is None:
        warnings.append("No decryptable schedule element was found at the legacy DRG schedule position.")
    if image_bytes is None:
        warnings.append("No decryptable image element was found at the legacy DRG image position.")

    return DrgPackage(
        source_path=path,
        source_bytes=data,
        text_encoding=encoding,
        header=header,
        elements=elements,
        schedule_text=schedule_text,
        image_bytes=image_bytes,
        metadata=metadata,
        warnings=warnings,
    )


def preserve_drg_package(package: DrgPackage, destination: str | Path) -> Path:
    root = Path(destination).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    source_name = package.source_path.name or "source.drg"
    source_destination = root / source_name
    source_destination.write_bytes(package.source_bytes)

    elements_dir = root / "elements"
    elements_dir.mkdir(exist_ok=True)
    manifest_elements: list[dict[str, Any]] = []
    for element in package.elements:
        suffix = _extension_for(element.media_type)
        stored_name = f"{element.index:02d}-{element.role}{suffix}"
        stored_path = elements_dir / stored_name
        stored_path.write_bytes(element.effective_bytes)
        entry = element.to_package_element(stored_name=f"elements/{stored_name}")
        manifest_elements.append({
            "index": entry.index,
            "role": entry.role,
            "size": entry.size,
            "sha256": entry.sha256,
            "encoding": entry.encoding,
            "media_type": entry.media_type,
            "text_preview": entry.text_preview,
            "stored_name": entry.stored_name,
            "encoded_sha256": sha256_bytes(element.encoded_text.encode(package.text_encoding, errors="replace")),
            "decoded_sha256": sha256_bytes(element.decoded_bytes),
            "decrypted": element.decrypted_bytes is not None,
        })

    if package.schedule_text is not None:
        (root / "schedule.sbg").write_text(package.schedule_text, encoding="utf-8")
    if package.image_bytes is not None:
        (root / f"image{_extension_for(_sniff_media_type(package.image_bytes))}").write_bytes(package.image_bytes)

    manifest = {
        "schema": "pysbagen.drg-preservation.v1",
        "source": {
            "original_path": str(package.source_path),
            "stored_name": source_name,
            "size": len(package.source_bytes),
            "sha256": package.source_sha256,
            "container_encoding": package.text_encoding,
        },
        "header": package.header,
        "metadata": package.metadata,
        "warnings": package.warnings,
        "elements": manifest_elements,
        "schedule_stored": package.schedule_text is not None,
        "image_stored": package.image_bytes is not None,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _decode_container_text(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def _legacy_role(index: int) -> str:
    if index == 2:
        return "image"
    if index == 4:
        return "schedule"
    return f"metadata-{index}"


def _decode_nested_base64(data: bytes) -> bytes:
    compact = b"".join(data.split())
    try:
        return base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return data


def _decode_schedule(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def _parse_metadata(data: bytes) -> Any | None:
    preview = _text_preview(data, limit=4096)
    if preview is None:
        return None
    stripped = preview.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    pairs: dict[str, str] = {}
    for line in stripped.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip():
                pairs[key.strip()] = value.strip()
    return pairs or stripped


def _text_preview(data: bytes, limit: int = 240) -> str | None:
    if not data:
        return ""
    sample = data[:limit]
    if b"\x00" in sample:
        return None
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = sample.decode("latin-1")
        except UnicodeDecodeError:
            return None
    printable = sum(character.isprintable() or character in "\r\n\t" for character in text)
    if printable / max(len(text), 1) < 0.85:
        return None
    return text


def _sniff_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if _text_preview(data) is not None:
        return "text/plain"
    return "application/octet-stream"


def _extension_for(media_type: str | None) -> str:
    explicit = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "audio/wav": ".wav",
        "text/plain": ".txt",
    }
    return explicit.get(media_type) or mimetypes.guess_extension(media_type or "") or ".bin"
