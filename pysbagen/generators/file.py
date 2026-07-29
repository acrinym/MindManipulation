from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .base import GenBase


def _ffmpeg_command(path: Path, sample_rate: int) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Decoding {path.suffix or 'this audio format'} requires FFmpeg to be installed and available on PATH"
        )
    return [
        ffmpeg,
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "2",
        "-ar", str(sample_rate),
        "pipe:1",
    ]


def _decode_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    result = subprocess.run(_ffmpeg_command(path, sample_rate), capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg could not decode {path.name}: {detail or 'unknown error'}")
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size % 2:
        raise ValueError(f"FFmpeg returned malformed stereo audio for {path.name}")
    return samples.reshape((-1, 2)).copy()


def _stereo_data(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Audio data must be one- or two-dimensional, got {data.shape}")
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def load_audio(path_value: str | Path, sample_rate: int = 44100) -> np.ndarray:
    """Load an entire file for callers that explicitly need a complete array."""
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, TypeError, ValueError):
        return _decode_with_ffmpeg(path, sample_rate)

    data = _stereo_data(data)
    if rate != sample_rate:
        factor = gcd(rate, sample_rate)
        data = resample_poly(
            data,
            sample_rate // factor,
            rate // factor,
            axis=0,
        ).astype(np.float32)
    return data.astype(np.float32, copy=False)


def _iter_array(
    data: np.ndarray,
    total_frames: int,
    frame_size: int,
    *,
    loop: bool,
) -> Iterator[np.ndarray]:
    if len(data) == 0:
        raise ValueError("Audio source is empty")
    produced = 0
    while produced < total_frames:
        take = min(frame_size, total_frames - produced)
        if loop:
            indices = (np.arange(take) + produced) % len(data)
            chunk = data[indices]
        else:
            chunk = data[produced : produced + take]
            if len(chunk) == 0:
                break
        produced += len(chunk)
        yield chunk.astype(np.float32, copy=False)
        if not loop and len(chunk) < take:
            break


def _iter_ffmpeg(
    path: Path,
    total_frames: int,
    frame_size: int,
    sample_rate: int,
    *,
    loop: bool,
) -> Iterator[np.ndarray]:
    produced = 0
    bytes_per_frame = 2 * np.dtype("<f4").itemsize
    while produced < total_frames:
        process = subprocess.Popen(
            _ffmpeg_command(path, sample_rate),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("Could not open FFmpeg audio pipes")
        decoded_this_pass = 0
        try:
            while produced < total_frames:
                take = min(frame_size, total_frames - produced)
                wanted = take * bytes_per_frame
                payload = bytearray()
                while len(payload) < wanted:
                    block = process.stdout.read(wanted - len(payload))
                    if not block:
                        break
                    payload.extend(block)
                usable = len(payload) - (len(payload) % bytes_per_frame)
                if usable == 0:
                    break
                samples = np.frombuffer(memoryview(payload)[:usable], dtype="<f4")
                chunk = samples.reshape((-1, 2)).copy()
                produced += len(chunk)
                decoded_this_pass += len(chunk)
                yield chunk
                if len(chunk) < take:
                    break
        finally:
            if process.poll() is None:
                if produced >= total_frames:
                    process.terminate()
                try:
                    _, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    _, stderr = process.communicate()
            else:
                stderr = process.stderr.read()

        if process.returncode not in (0, -15) and produced < total_frames:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"FFmpeg could not decode {path.name}: {detail or 'unknown error'}")
        if decoded_this_pass == 0:
            raise ValueError(f"Audio file is empty or undecodable: {path}")
        if not loop:
            break


def iter_audio_chunks(
    path_value: str | Path,
    total_frames: int,
    frame_size: int,
    sample_rate: int = 44100,
    *,
    loop: bool = False,
) -> Iterator[np.ndarray]:
    """Stream decoded stereo chunks with bounded memory, restarting only when looping."""
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if total_frames <= 0:
        return

    source: sf.SoundFile | None = None
    try:
        source = sf.SoundFile(path)
    except (RuntimeError, TypeError, ValueError):
        yield from _iter_ffmpeg(path, total_frames, frame_size, sample_rate, loop=loop)
        return

    with source:
        if source.samplerate != sample_rate:
            if shutil.which("ffmpeg") is not None:
                yield from _iter_ffmpeg(path, total_frames, frame_size, sample_rate, loop=loop)
                return
            data = load_audio(path, sample_rate)
            yield from _iter_array(data, total_frames, frame_size, loop=loop)
            return

        produced = 0
        decoded_any = False
        while produced < total_frames:
            take = min(frame_size, total_frames - produced)
            chunk = source.read(take, dtype="float32", always_2d=True)
            chunk = _stereo_data(chunk)
            if len(chunk) == 0:
                if loop and decoded_any:
                    source.seek(0)
                    continue
                break
            decoded_any = True
            produced += len(chunk)
            yield chunk.astype(np.float32, copy=False)
        if not decoded_any:
            raise ValueError(f"Audio file is empty: {path}")


@dataclass
class FileSpec(GenBase):
    path: str = ""
    loop: bool = False

    def _load(self) -> np.ndarray:
        return load_audio(self.path, self.sample_rate)

    def generator(self, duration: float):
        total_frames = max(0, int(self.sample_rate * duration))
        for chunk in iter_audio_chunks(
            self.path,
            total_frames,
            self.frame,
            self.sample_rate,
            loop=self.loop,
        ):
            yield (chunk * self._amp_scale()).astype(np.float32), {
                "type": "file",
                "path": Path(self.path).name,
            }
