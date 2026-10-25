# Local Qwen ASR

## Experimental FlashAttention 2 branch

This branch adds `--attention flash_attention_2`; SDPA remains the default. It pins
PyTorch 2.10.0/CUDA 12.8 and Hugging Face `kernels` 0.17.2. Use the isolated setup
below rather than `setup.cmd` if you want to preserve the existing `.venv`:

```powershell
.\setup-flash.cmd
.\.local\venv-flash\Scripts\python.exe -m qwen_asr_cli "recording.mp4" --attention flash_attention_2
```

The Windows build is pinned to `kernels-community/flash-attn2` model-repository
revision `c3f3c93a4a39e3d2cf49c13eac49229bc726ea13`. The newer kernel-type repository
does not carry that Windows build. This branch downloads only the matching
`torch210-cu128-x86_64-windows` build and uses the kernels library's local override.
Its older metadata is adapted in a separate local copy for the current loader;
the CUDA binary is unchanged and its SHA-256 is checked against the pinned Hub artifact.
Run from the checkout's drive; the local-override syntax requires a relative Windows path.
This is executable third-party kernel code, not a model weight. SDPA does not load it.
Use `--offline` after the model and kernel have both been cached.

`scripts/benchmark_attention.py` prepares local audio and compares selectable backends
with identical chunking, BF16, greedy generation, and token limits. Results and private
transcripts go only under `.local/benchmark-attention`. It reports decoding, model
loading, warmed inference, peak GPU allocation, and observed attention operators
separately. A changed transcript is not proof of better or worse accuracy without a
human reference. Do not compare different PyTorch versions as if only attention changed.

Native Windows CLI for local Russian/English audio and video transcription with Qwen3-ASR 1.7B. Uses an NVIDIA CUDA GPU. No WSL, Docker, web server, or cloud transcription.

## Setup

Prerequisites: Windows x64, a BF16-capable NVIDIA GPU, a driver compatible with CUDA 12.8, and current `uv`, `ffmpeg`, and `ffprobe` executables on PATH. Tested with an RTX 4090 (24 GB VRAM); minimum VRAM has not been established. Setup downloads project-local Python 3.12.15 and the locked CUDA-enabled Python dependencies. Model weights download on first transcription; timestamps require an additional aligner model.

Use this repository as a checkout and run its setup script. Standalone wheel/pip installation and other operating systems are not supported installation paths.

```powershell
.\setup.cmd
```

The setup wrapper uses a process-local PowerShell policy bypass, not a permanent policy change. Its equivalent command is:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

No global Python packages, PATH changes, Linux components, or separate CUDA Toolkit are required. Keep the project in its original location: moving a Python virtual environment requires recreating it.

## Transcribe

```powershell
.\qwen-asr.cmd "D:\Recordings\meeting.mp4"
.\qwen-asr.cmd "D:\Recordings\meeting.mp4" --timestamps
.\qwen-asr.cmd "D:\Recordings\interview.wav" --language ru
.\qwen-asr.cmd "D:\Recordings\meeting.mp4" --output "D:\Transcripts\meeting.txt"
.\qwen-asr.cmd "D:\Recordings\meeting.mp4" --stdout
.\qwen-asr.cmd "D:\Recordings\meeting.mp4" --offline --force
.\qwen-asr.cmd --help
```

The installed console entry point also works without activating the environment:

```powershell
.\.venv\Scripts\qwen-asr.exe "D:\Recordings\meeting.mp4"
```

- One local audio or video file per command. No URLs or folder batches.
- Default output is a UTF-8 `.txt` file beside the input. Existing files are protected unless `--force` is supplied. The source media is never a valid output target.
- `--stdout` writes only the transcript to stdout. Progress and errors go to stderr. Use `--output` rather than shell redirection when consistent UTF-8 file encoding matters, particularly in Windows PowerShell 5.
- Language detection is automatic. Use `--language ru` or `--language en` for monolingual recordings. Mixed-language input should normally use automatic detection. Output is transcription, not translation.
- `--timestamps` adds aligned segment ranges, using the official forced aligner. It does not add speaker identities or produce subtitles.
- `--offline` forbids model downloads and uses cached model files. Run each desired mode online once first. Model downloads are the only intentional network use; audio is decoded and processed locally.
- CUDA is required. The program does not silently fall back to CPU.
- Ctrl+C cancels the run. Final files are written only after successful processing; temporary decoded audio is cleaned up.

Exit codes: `0` success; `2` input, argument, or file I/O error; `3` audio decoding error; `4` model/GPU setup error; `5` inference/alignment error; `130` cancellation. stdout delivery cannot be retracted if interrupted while piping the finished transcript.

`--chunk-seconds` controls the maximum audio chunk (default 30, range 1–60). `--max-new-tokens` controls the per-chunk generation cap (default 1024, range 1–8192). Lower chunk sizes reduce context and can increase cut-word errors. Timestamp lines use `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`.

Displayed milliseconds are formatting, not a claim of millisecond alignment accuracy. The aligner uses approximately 80-ms timestamp steps.

Only the first audio track is transcribed, downmixed to mono. Times are relative to the container's start, not the first spoken word. Initial audio delay and internal timestamp gaps become silence, with 1-ms jitter tolerance. Trailing video without audio is not padded. FFmpeg/FFprobe cannot fetch network media through playlists.

## Storage

Python, dependencies, model downloads, caches, and working files stay in this checkout. Final transcripts go to the requested output location, which defaults to beside the input recording.

| Path | Contents |
|---|---|
| `.venv/` | Project virtual environment |
| `.local/python/` | Managed Python installation |
| `.local/uv-cache/` | Package download/build cache |
| `.local/models/` | Hugging Face model files and related cache |
| `.local/tmp/` | Temporary decoded audio and process temporary files |
| `.local/fixtures/` | Public test audio and source information |

These directories are ignored by Git. Other runtime caches are also redirected beneath `.local/`. Existing `uv`, FFmpeg, the NVIDIA driver, and OS-managed logs/caches remain outside the project. Do not delete `.local/python` while retaining `.venv`; the environment depends on it.

Keep private recordings and transcripts under `.local/` or outside the repository. `/recordings/`, `/transcripts/`, common media/model files, private-key files, and `.env` files are also ignored. Arbitrarily named plain-text transcripts elsewhere in the checkout are not automatically ignored. Review `git diff --cached` before committing; ignore rules do not protect files already tracked or added with `--force`.

Audio is decoded to temporary mono 16 kHz PCM, roughly 115 MB per hour. Allow additional disk space for models, CUDA dependencies, and package caches. Processing uses bounded audio chunks rather than loading an hours-long recording into GPU memory.

## Accuracy and limitations

Recognition and alignment can be wrong, especially with overlapping speakers, names, noise, silence, and language changes. The CLI splits at substantial pauses (at least 400 ms after roughly five seconds of context), even when a file is shorter than the maximum chunk. This reduces cross-language translation errors observed when Qwen receives Russian and English together. Rapid language switches without pauses can still be mistranscribed or translated by the model; inspect such passages.

Silence-aware boundaries reduce cut-word errors but cannot eliminate them. With no suitable pause near the maximum chunk length, a hard boundary is necessary. Chunking also limits context between passages. Silence skipping detects near-digital silence, not all background noise. This is not a verbatim-transcription guarantee.

Generation that reaches its output-token limit is treated as an error, not silently saved as a successful truncated transcript. Timestamp alignment is additional inference and may fail on poorly recognized speech. No speaker diarization or LLM cleanup is performed.

## Development

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\scripts\environment.ps1
uv sync --locked
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

Ordinary tests generate their own audio and mock the model APIs; they do not require downloaded model weights, a GPU, or pre-existing `.local` contents. FFmpeg integration tests require the documented FFmpeg/FFprobe prerequisites. Verification on the tested RTX 4090 produced **121 passed tests, 2 Windows symlink-privilege skips, and 23 passed end-to-end acceptance cases**, including real RU/EN transcription and alignment. Setup re-run and lockfile synchronization also passed.

### Maintainer GPU acceptance checks

The scripts below require a separately prepared `.local/fixtures` directory containing `en_fleurs.wav`, `ru_fleurs.wav`, `mixed_ru_en.wav`, and the reference/provenance `manifest.json`. Those assets and local reports are deliberately not distributed in this repository. These are maintainer checks, not a self-contained clean-clone test command.

With those fixtures prepared, prefetch both pinned models before running the offline acceptance checks:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe scripts\prepare_acceptance_media.py
.\.venv\Scripts\python.exe scripts\verify_gpu.py
```

This writes transcripts, logs, timings, and normalized word-error rates under `.local/verification`. The fixture manifest under `.local/fixtures` records Google FLEURS sources, reference text, hashes, and CC BY 4.0 attribution. Fixture downloads are not part of the ordinary unit-test run. The two-hour fixture is silence: it checks bounded decoding/chunking, not two-hour speech accuracy.

## Licensing

No source-code license has been specified for this project. The upstream Qwen model weights are distributed separately under Apache 2.0. The public Google FLEURS samples used for maintainer verification are CC BY 4.0; they are not bundled here.

## Sources

- [Official native Transformers ASR checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- [Official native forced aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/)
