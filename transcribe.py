import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import fnmatch
import torch
import whisper


SUPPORTED_EXTS = {
    ".mp3", ".mp4", ".m4a", ".aac", ".wav",
    ".flac", ".ogg", ".mkv", ".webm", ".mov"
}


def list_media_files(path: Path, glob_pattern: Optional[str]) -> List[Path]:
    """Return a list of media files.

    - If path is a file: return [path] (ignoring glob_pattern).
    - If path is a directory:
        * If glob_pattern is provided, use rglob(glob_pattern).
        * Otherwise, rglob all files and filter by SUPPORTED_EXTS.
    """
    if path.is_file():
        return [path]

    files: List[Path] = []
    if glob_pattern:
        # Use rglob with the user pattern, then filter to supported extensions
        for p in sorted(path.rglob(glob_pattern)):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                files.append(p)
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                files.append(p)
    return files


def pick_device(prefer_gpu: bool) -> Tuple[str, dict]:
    info = {}
    if prefer_gpu and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        info["device_name"] = device_name
        info["device_type"] = "cuda"
        return "cuda", info
    info["device_type"] = "cpu"
    return "cpu", info


def srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def iso_timestamp(seconds: float, md_date: Optional[str]) -> str:
    """Return ISO 8601-2 style timestamp.

    If md_date is provided (YYYY-MM-DD), returns e.g. '2025-12-02T00:03:27Z'.
    Otherwise returns time-only 'HH:MM:SS' which is still ISO 8601 compliant.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if md_date:
        return f"{md_date}T{h:02d}:{m:02d}:{s:02d}Z"
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_file(model, file_path: Path, language: Optional[str], fp16: bool, verbose_json: bool, temperature: float):
    t0 = time.time()
    result = model.transcribe(
        str(file_path),
        language=language,
        fp16=fp16,
        verbose=verbose_json,
        temperature=temperature,
    )
    t1 = time.time()
    elapsed = t1 - t0

    audio_dur = result.get("duration", None)
    tokens_total = 0
    if verbose_json:
        for seg in result.get("segments", []):
            if "tokens" in seg and isinstance(seg["tokens"], list):
                tokens_total += len(seg["tokens"])

    metrics = {
        "elapsed_s": elapsed,
        "audio_duration_s": audio_dur,
        "rtf": (elapsed / audio_dur) if audio_dur else None,
        "tokens_total": tokens_total if tokens_total else None,
        "tokens_per_s": (tokens_total / elapsed) if tokens_total else None,
    }
    return result, metrics


def write_outputs(result, file_path: Path, out_dir: Path, md_date: Optional[str] = None):
    stem = file_path.stem
    safe_stem = stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / safe_stem

    # Plain text
    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(result.get("text", "").strip() + "\n")

    segs = result.get("segments", [])

    # SRT
    with open(base.with_suffix(".srt"), "w", encoding="utf-8") as f:
        for i, seg in enumerate(segs, start=1):
            start = srt_timestamp(seg["start"])
            end = srt_timestamp(seg["end"])
            text = seg.get("text", "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    # VTT
    with open(base.with_suffix(".vtt"), "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segs:
            start = srt_timestamp(seg["start"]).replace(",", ".")
            end = srt_timestamp(seg["end"]).replace(",", ".")
            text = seg.get("text", "").strip()
            f.write(f"{start} --> {end}\n{text}\n\n")

    # Markdown with ISO 8601-style timestamps
    with open(base.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(f"# {safe_stem}\n\n")
        for seg in segs:
            ts = iso_timestamp(seg["start"], md_date)
            text = seg.get("text", "").strip()
            if not text:
                continue
            f.write(f"- {ts} — {text}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ROCm-accelerated Whisper transcriber (auto GPU/CPU) with Markdown export and optional glob filtering."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to media file or directory containing media files."
    )
    parser.add_argument(
        "--out_dir", "-o", type=str, default="/data/out",
        help="Output directory (default: /data/out)."
    )
    parser.add_argument(
        "--model", "-m", type=str, default="medium",
        help="Whisper model size (tiny, base, small, medium, large, etc.)."
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None,
        help="Force language code (e.g., en). Default: auto-detect."
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Prefer GPU if available (ROCm)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)."
    )
    parser.add_argument(
        "--verbose_json", action="store_true",
        help="Return verbose JSON to compute token throughput."
    )
    parser.add_argument(
        "--md_date", type=str, default=None,
        help="ISO 8601-2 date to prefix timestamps (YYYY-MM-DD). If omitted, timestamps are HH:MM:SS only."
    )
    parser.add_argument(
        "--glob", type=str, default=None,
        help="Optional glob pattern (e.g. '*.mkv') when --input is a directory. "
             "If provided, only matching files with supported extensions are processed."
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    assert in_path.exists(), f"Input path not found: {in_path}"

    device, dinfo = pick_device(prefer_gpu=args.gpu)
    print(f"[info] device={device} details={dinfo}")

    print(f"[info] loading model '{args.model}' ...")
    model = whisper.load_model(args.model, device=device)
    fp16 = (device == "cuda")

    files = list_media_files(in_path, args.glob)
    if not files:
        if in_path.is_dir() and args.glob:
            print(f"[warn] no media files found in {in_path} matching glob '{args.glob}'.")
        else:
            print(f"[warn] no media files found in {in_path}.")
        return 0

    print(f"[info] found {len(files)} media file(s).")
    for f in files:
        print(f"[start] {f}")
        t_start = time.time()
        result, metrics = transcribe_file(model, f, args.language, fp16, args.verbose_json, args.temperature)
        write_outputs(result, f, out_dir, md_date=args.md_date)
        t_end = time.time()

        elapsed = t_end - t_start
        dur = metrics.get("audio_duration_s")
        rtf = metrics.get("rtf")
        tps = metrics.get("tokens_per_s")
        toks = metrics.get("tokens_total")
        rtf_str = f"{rtf:.3f}" if rtf is not None else "n/a"
        tps_str = f"{tps:.1f}" if tps is not None else "n/a"
        print(f"[done] {f.name} elapsed={elapsed:.2f}s audio_dur={dur if dur else 'n/a'}s rtf={rtf_str} tokens={toks} tps={tps_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
