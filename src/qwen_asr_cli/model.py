from dataclasses import dataclass
from typing import TYPE_CHECKING

from .alignment import Word
from .audio import SAMPLE_RATE
from .errors import ModelError, TranscriptionError
from .runtime import configure_runtime

if TYPE_CHECKING:
    import numpy as np

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
ASR_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGNER_REVISION = "c07281df297b9905d24a508279258cccf987a064"
FLASH_KERNEL = "kernels-community/flash-attn2@c3f3c93a4a39e3d2cf49c13eac49229bc726ea13"
ALIGNER_LANGUAGES = {
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
}


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None


def check_generation_limit(generated_count: int, limit: int) -> None:
    if generated_count >= limit:
        raise TranscriptionError(
            f"Generation reached --max-new-tokens={limit}; the transcript may be truncated. "
            "Increase that limit or reduce --chunk-seconds. No final transcript was saved."
        )


def alignment_language(language: str | None) -> str | None:
    if language is None or language in ALIGNER_LANGUAGES:
        return language
    parts = {part.strip().casefold() for part in language.split(",")}
    if parts <= {"russian", "english"}:
        return None
    raise TranscriptionError(
        f"The forced aligner does not support detected language {language!r}. "
        "Retry without --timestamps."
    )


class NativeQwen:
    def __init__(self, offline: bool = False, progress=lambda message: None, attention="sdpa"):
        if attention not in {"sdpa", "flash_attention_2"}:
            raise ModelError(f"Unsupported attention backend: {attention}")
        self.attention = FLASH_KERNEL if attention == "flash_attention_2" else "sdpa"
        self.local = configure_runtime(offline)
        if attention == "flash_attention_2":
            from .flash import prepare_flash_kernel

            try:
                prepare_flash_kernel(offline)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise ModelError(f"FlashAttention kernel preparation failed: {exc}") from exc
        self.offline = offline
        self.progress = progress
        self.aligner = None
        self.aligner_processor = None
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ModelError(f"Missing model dependency; run setup.cmd again: {exc}") from exc
        self.torch = torch
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                raise ModelError(
                    "CUDA is unavailable. Use the project CUDA PyTorch environment "
                    "and an NVIDIA driver."
                )
            if not torch.cuda.is_bf16_supported():
                raise ModelError(
                    "This GPU does not support BF16. This installation targets the RTX 4090."
                )
        except (RuntimeError, AssertionError) as exc:
            raise ModelError(f"CUDA initialization failed: {exc}") from exc
        suffix = " from cache only" if offline else ""
        progress(f"Loading {ASR_MODEL} on CUDA (BF16, {attention}){suffix}…")
        try:
            self.processor = AutoProcessor.from_pretrained(ASR_MODEL, **self._options(ASR_REVISION))
            self.model = AutoModelForMultimodalLM.from_pretrained(
                ASR_MODEL, **self._options(ASR_REVISION), dtype=torch.bfloat16,
                device_map="cuda", attn_implementation=self.attention,
            ).eval()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise ModelError(self._load_error(ASR_MODEL, exc)) from exc

    def _options(self, revision: str) -> dict:
        return {
            "revision": revision, "local_files_only": self.offline,
            "cache_dir": str(self.local / "models" / "hub"), "trust_remote_code": False,
        }

    def _load_error(self, model: str, error: Exception) -> str:
        hint = " Run once without --offline to download the pinned model." if self.offline else ""
        return f"Could not load {model}.{hint} {error}"

    def transcribe(
        self, samples: "np.ndarray", language: str | None, max_new_tokens: int
    ) -> Transcript:
        try:
            inputs = self.processor.apply_transcription_request(
                audio=samples, language=language,
                processor_kwargs={"sampling_rate": SAMPLE_RATE},
            ).to(self.model.device, self.model.dtype)
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, disable_compile=True,
                )
            generated = output[0, inputs["input_ids"].shape[1] :]
            check_generation_limit(len(generated), max_new_tokens)
            parsed = self.processor.decode(generated, return_format="parsed")
            text = " ".join(parsed["transcription"].split())
            return Transcript(text, language or parsed["language"])
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise TranscriptionError(f"ASR inference failed: {exc}") from exc

    def _load_aligner(self) -> None:
        if self.aligner is not None:
            return
        self.progress(f"Loading {ALIGNER_MODEL} on CUDA (BF16, {self.attention})…")
        try:
            from transformers import AutoModelForTokenClassification, AutoProcessor

            self.aligner_processor = AutoProcessor.from_pretrained(
                ALIGNER_MODEL, **self._options(ALIGNER_REVISION),
            )
            self.aligner = AutoModelForTokenClassification.from_pretrained(
                ALIGNER_MODEL, **self._options(ALIGNER_REVISION), dtype=self.torch.bfloat16,
                device_map="cuda", attn_implementation=self.attention,
            ).eval()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise ModelError(self._load_error(ALIGNER_MODEL, exc)) from exc

    def align(self, samples: "np.ndarray", transcript: Transcript) -> list[Word]:
        language = alignment_language(transcript.language)
        self._load_aligner()
        try:
            inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
                audio=samples, transcript=transcript.text, language=language,
                processor_kwargs={"sampling_rate": SAMPLE_RATE},
            )
            if not word_lists[0]:
                raise TranscriptionError(
                    "The transcript contains no alignable words. Retry without --timestamps."
                )
            inputs = inputs.to(self.aligner.device, self.aligner.dtype)
            with self.torch.inference_mode():
                output = self.aligner(**inputs)
            decoded = self.aligner_processor.decode_forced_alignment(
                logits=output.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
                timestamp_token_id=self.aligner.config.timestamp_token_id,
            )[0]
            if [item["text"] for item in decoded] != word_lists[0]:
                raise TranscriptionError("The forced aligner did not return every transcript word.")
            return [
                Word(item["text"], float(item["start_time"]), float(item["end_time"]))
                for item in decoded
            ]
        except (ImportError, OSError, RuntimeError, ValueError, IndexError) as exc:
            raise TranscriptionError(
                f"Forced alignment failed; retry without --timestamps: {exc}"
            ) from exc
