#!/usr/bin/env python3
"""
System Audio Transcriber  —  cross-platform (Windows / Linux / macOS)
Captures system audio + microphone, transcribes in real-time to English,
and saves to a session-based txt file.

Usage:
    python transcribe.py [--chunk-sec 5] [--model base] [--output-dir ./sessions]
"""

import argparse
import platform
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed. Run: pip install faster-whisper")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono
CHANNELS = 1
OS = platform.system()  # "Windows", "Linux", "Darwin"

# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------


def new_session(output_dir: Path) -> tuple[str, Path]:
    session_id = str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)
    session_file = output_dir / f"{session_id}.txt"
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_file.write_text(
        f"Session ID : {session_id}\nStarted    : {started_at}\n{'─' * 60}\n\n",
        encoding="utf-8",
    )
    return session_id, session_file


def append_transcript(session_file: Path, text: str, timestamp: str) -> None:
    with session_file.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------


def list_devices() -> None:
    """Print all audio devices with their index and host API."""
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    print(f"\nAvailable audio devices (OS: {OS}):\n")
    print(f"  {'Idx':>3}  {'I':>2}  {'O':>2}  {'Host API':<14}  Name")
    print(
        f"  {'───':>3}  {'──':>2}  {'──':>2}  {'──────────────':<14}  ────────────────────────────"
    )
    for i, dev in enumerate(devices):
        api_name = (
            host_apis[dev["hostapi"]]["name"]
            if dev["hostapi"] < len(host_apis)
            else "?"
        )
        print(
            f"  {i:>3}  {dev['max_input_channels']:>2}  {dev['max_output_channels']:>2}"
            f"  {api_name:<14}  {dev['name']}"
        )
    print()


def _find_wasapi_loopback() -> int | None:
    """
    Windows: find the default output device that supports WASAPI loopback.
    Returns the output device index to be opened with wasapi_loopback=True.
    """
    host_apis = sd.query_hostapis()
    wasapi_idx = next(
        (i for i, api in enumerate(host_apis) if "wasapi" in api["name"].lower()), None
    )
    if wasapi_idx is None:
        return None

    devices = sd.query_devices()
    # Prefer the default output device for WASAPI
    try:
        default_out = sd.default.device[1]
        dev = devices[default_out]
        if dev["hostapi"] == wasapi_idx and dev["max_output_channels"] > 0:
            return default_out
    except Exception:
        pass

    # Fall back: first WASAPI output device
    for i, dev in enumerate(devices):
        if dev["hostapi"] == wasapi_idx and dev["max_output_channels"] > 0:
            return i
    return None


def _find_linux_monitor() -> int | None:
    """
    Linux: find a PulseAudio/PipeWire monitor source.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_input_channels"] > 0 and ("monitor" in name or "loopback" in name):
            return i
    return None


def _find_macos_loopback() -> int | None:
    """
    macOS: find a virtual loopback device (BlackHole, Soundflower, Loopback).
    """
    devices = sd.query_devices()
    keywords = ("blackhole", "soundflower", "loopback", "virtual")
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_input_channels"] > 0 and any(k in name for k in keywords):
            return i
    return None


def find_system_audio_source() -> tuple[int | None, bool]:
    """
    Auto-detect system audio source.
    Returns (device_index, is_wasapi_loopback).
    """
    if OS == "Windows":
        idx = _find_wasapi_loopback()
        return idx, True
    elif OS == "Linux":
        return _find_linux_monitor(), False
    elif OS == "Darwin":
        return _find_macos_loopback(), False
    return None, False


def open_sys_stream(
    device: int,
    wasapi_loopback: bool,
    callback,
) -> sd.InputStream:
    """Open the system audio input stream, using WASAPI loopback on Windows."""
    kwargs: dict = dict(
        device=device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    )
    if wasapi_loopback:
        # sounddevice exposes WASAPI extras via extra_settings
        try:
            import sounddevice as _sd

            wasapi_settings = _sd.WasapiSettings(loopback=True)
            kwargs["extra_settings"] = wasapi_settings
        except AttributeError:
            # Older sounddevice — try low-level approach
            kwargs["extra_settings"] = None
    return sd.InputStream(**kwargs)


def mix_streams(mic_data: np.ndarray, sys_data: np.ndarray) -> np.ndarray:
    """Average two mono float32 arrays, handling length mismatch."""
    min_len = min(len(mic_data), len(sys_data))
    if min_len == 0:
        return (
            mic_data.astype(np.float32)
            if len(mic_data)
            else sys_data.astype(np.float32)
        )
    return (
        mic_data[:min_len].astype(np.float32) + sys_data[:min_len].astype(np.float32)
    ) / 2.0


# ---------------------------------------------------------------------------
# Transcription worker
# ---------------------------------------------------------------------------


class TranscribeWorker(threading.Thread):
    def __init__(
        self,
        audio_queue: "queue.Queue[np.ndarray]",
        model: WhisperModel,
        session_file: Path,
        verbose: bool = True,
    ):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.model = model
        self.session_file = session_file
        self.verbose = verbose
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk: np.ndarray = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            segments, _info = self.model.transcribe(
                chunk,
                task="translate",  # always output English
                language=None,  # auto-detect
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if not text_parts:
                continue

            full_text = " ".join(text_parts)
            ts = datetime.now().strftime("%H:%M:%S")
            append_transcript(self.session_file, full_text, ts)

            if self.verbose:
                print(f"  [{ts}] {full_text}", flush=True)


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    if args.list_devices:
        list_devices()
        return

    output_dir = Path(args.output_dir)
    session_id, session_file = new_session(output_dir)

    print(f"\n{'═' * 60}")
    print(f"  OS         : {OS}")
    print(f"  Session ID : {session_id}")
    print(f"  Output     : {session_file}")
    print(f"  Model      : {args.model}")
    print(f"  Chunk size : {args.chunk_sec}s")
    print(f"{'═' * 60}")
    print("  Loading Whisper model …", end=" ", flush=True)

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )
    print("ready.")

    # ── Resolve system audio device ─────────────────────────────────────────
    wasapi_loopback = False
    if args.sys_device is not None:
        sys_device = args.sys_device
        wasapi_loopback = (
            OS == "Windows"
        )  # assume WASAPI loopback if user specified on Windows
    else:
        sys_device, wasapi_loopback = find_system_audio_source()
        if sys_device is not None:
            dev_name = sd.query_devices(sys_device)["name"]
            mode = "WASAPI loopback" if wasapi_loopback else "monitor source"
            print(f"  Auto-detected system audio [{mode}]: [{sys_device}] {dev_name}")
        else:
            print(
                "  Warning: could not auto-detect a system audio source.\n"
                + _no_sys_audio_hint()
            )

    print("  Recording … Press Ctrl+C to stop.\n")

    mic_device = args.mic_device
    chunk_frames = int(SAMPLE_RATE * args.chunk_sec)
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

    mic_buffer: list[np.ndarray] = []
    sys_buffer: list[np.ndarray] = []
    buffer_lock = threading.Lock()

    def mic_callback(indata, frames, time_info, status):
        with buffer_lock:
            mic_buffer.append(indata[:, 0].copy())

    def sys_callback(indata, frames, time_info, status):
        with buffer_lock:
            sys_buffer.append(indata[:, 0].copy())

    stop_flag = threading.Event()

    def chunk_feeder():
        while not stop_flag.is_set():
            time.sleep(args.chunk_sec)
            with buffer_lock:
                mic_data = (
                    np.concatenate(mic_buffer)
                    if mic_buffer
                    else np.zeros(chunk_frames, dtype=np.float32)
                )
                sys_data = (
                    np.concatenate(sys_buffer)
                    if sys_buffer
                    else np.array([], dtype=np.float32)
                )
                mic_buffer.clear()
                sys_buffer.clear()

            if sys_device is not None and len(sys_data) > 0:
                chunk = mix_streams(mic_data, sys_data)
            else:
                chunk = mic_data.astype(np.float32)

            max_val = np.abs(chunk).max()
            if max_val > 0:
                chunk = chunk / max_val

            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    worker = TranscribeWorker(audio_queue, model, session_file, verbose=not args.quiet)
    worker.start()

    feeder = threading.Thread(target=chunk_feeder, daemon=True)
    feeder.start()

    # ── Open audio streams ──────────────────────────────────────────────────
    streams: list[sd.InputStream] = []

    mic_stream = sd.InputStream(
        device=mic_device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=mic_callback,
        blocksize=1024,
    )
    streams.append(mic_stream)

    if sys_device is not None:
        try:
            sys_stream = open_sys_stream(sys_device, wasapi_loopback, sys_callback)
            streams.append(sys_stream)
        except Exception as e:
            print(f"  Warning: could not open system audio stream: {e}")
            print("  Continuing with microphone only.")

    try:
        for s in streams:
            s.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n  Stopping … flushing remaining audio …")
    finally:
        stop_flag.set()
        for s in streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        feeder.join(timeout=args.chunk_sec + 2)
        worker.stop()
        worker.join(timeout=10)

        ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with session_file.open("a", encoding="utf-8") as f:
            f.write(f"\n{'─' * 60}\n")
            f.write(f"Ended      : {ended_at}\n")

        print(f"\n  Session saved to: {session_file}")


def _no_sys_audio_hint() -> str:
    if OS == "Windows":
        return (
            "  On Windows: make sure WASAPI is available (it is on all modern Windows).\n"
            "  Run --list-devices and pass the output device index with --sys-device <idx>.\n"
            "  Only microphone will be captured for now."
        )
    elif OS == "Linux":
        return (
            "  On Linux: enable a PulseAudio/PipeWire monitor source.\n"
            "  Run --list-devices and look for a device with 'monitor' in the name.\n"
            "  Only microphone will be captured for now."
        )
    elif OS == "Darwin":
        return (
            "  On macOS: install BlackHole (free) from https://existential.audio/blackhole/\n"
            "  then use Audio MIDI Setup to create a Multi-Output Device.\n"
            "  Only microphone will be captured for now."
        )
    return "  Only microphone will be captured for now."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time system audio + mic transcriber → English txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (larger = more accurate but slower)",
    )
    p.add_argument(
        "--chunk-sec",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Audio chunk duration sent to Whisper per inference",
    )
    p.add_argument(
        "--output-dir",
        default="./sessions",
        help="Directory to store session txt files",
    )
    p.add_argument(
        "--mic-device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Microphone device index (default: system default)",
    )
    p.add_argument(
        "--sys-device",
        type=int,
        default=None,
        metavar="INDEX",
        help="System audio device index (auto-detected if omitted)",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live transcript output to stdout",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
