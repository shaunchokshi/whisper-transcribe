# rocm_whisper_transcriber (Markdown + glob support)

ROCm-accelerated transcription for AMD GPUs using OpenAI Whisper + PyTorch ROCm inside Docker,
with:

- Automatic Markdown export including ISO 8601-style timestamps, and
- Optional glob-based filtering of which files to process.

## Quick start

```bash
unzip rocm_whisper_transcriber_md_glob.zip
cd rocm_whisper_transcriber_md_glob
bash run_rocm_whisper.sh
```

The script will:

1. Prompt for a file or directory path.
2. Prompt for an ISO date (YYYY-MM-DD) to embed in Markdown timestamps (optional).
3. Prompt for a glob pattern (e.g. `*.mkv`) when the input is a directory (optional).
4. Build and run the ROCm-enabled Whisper container.

Media is read from the host directory you choose, mounted at `/data`.
Outputs are written to `/data/out` (mapped to `./data/out` on the host).

## Glob behaviour

- If `--input` is a **file**, the glob setting is ignored and only that file is processed.
- If `--input` is a **directory**:
  - If you supply `--glob '*.mkv'`, the script recursively finds only files under that
    directory matching the pattern **and** having a supported extension.
  - If you leave glob blank, it processes all supported media types:

    ```text
    .mp3, .mp4, .m4a, .aac, .wav, .flac, .ogg, .mkv, .webm, .mov
    ```

This avoids relying on shell expansion inside Docker while still giving you wildcard control.

## Outputs

For each media file you get:

- `<stem>.txt`  – full plain-text transcript
- `<stem>.srt`  – subtitles
- `<stem>.vtt`  – WebVTT
- `<stem>.md`   – Markdown notes with timestamps

Example Markdown:

```markdown
# lecture_2025-12-01

- 2025-12-02T00:00:03Z — Intro to topic
- 2025-12-02T00:00:45Z — Definition of key term
```

- If you set `--md_date 2025-12-02` (or enter it at the prompt), timestamps are
  formatted as `YYYY-MM-DDTHH:MM:SSZ`.
- If you leave it blank, timestamps are `HH:MM:SS`.

The time component is the segment offset from the start of the recording.

## Manual example

```bash
docker compose build whisper

docker run --rm -it   --device=/dev/kfd --device=/dev/dri   --group-add video   -e HIP_VISIBLE_DEVICES=0   -e HSA_OVERRIDE_GFX_VERSION=10.3.0   -v "$PWD/data":/data   rocm-whisper:latest   python /app/transcribe.py     --gpu     --model small.en     --input /data     --glob "*.mkv"     --out_dir /data/out     --verbose_json     --md_date 2025-12-02
```
