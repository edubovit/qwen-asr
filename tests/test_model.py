import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_asr_cli import model
from qwen_asr_cli.alignment import Word
from qwen_asr_cli.errors import ModelError, TranscriptionError


@pytest.fixture
def fake_libraries(monkeypatch):
    calls = []

    class Batch(dict):
        def to(self, device, dtype):
            assert device == "cuda" and dtype == "bf16"
            return self

    class Processor:
        def apply_transcription_request(self, **kwargs):
            calls.append(("prepare_asr", kwargs))
            return Batch(input_ids=np.array([[5, 6]]))

        def decode(self, generated, return_format):
            assert generated.ndim == 1
            assert return_format == "parsed"
            return {"language": "English", "transcription": " Hello,  world! "}

        def prepare_forced_aligner_inputs(self, **kwargs):
            calls.append(("prepare_alignment", kwargs))
            return Batch(input_ids=np.array([[5, 6]])), [["Hello", "world"]]

        def decode_forced_alignment(self, **kwargs):
            calls.append(("decode_alignment", kwargs))
            return [[
                {"text": "Hello", "start_time": 0.1, "end_time": 0.5},
                {"text": "world", "start_time": 0.5, "end_time": 0.9},
            ]]

    class Model:
        device = "cuda"
        dtype = "bf16"
        config = SimpleNamespace(timestamp_token_id=123)

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return np.array([[5, 6, 20, 21, 151645]])

        def __call__(self, **kwargs):
            return SimpleNamespace(logits="logits")

    def factory(kind, cls):
        def load(name, **kwargs):
            calls.append((kind, {"name": name, **kwargs}))
            return cls()

        return SimpleNamespace(from_pretrained=load)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True,
        ),
        bfloat16="bf16", inference_mode=nullcontext,
    )
    transformers = SimpleNamespace(
        AutoProcessor=factory("processor", Processor),
        AutoModelForMultimodalLM=factory("asr_model", Model),
        AutoModelForTokenClassification=factory("aligner_model", Model),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return SimpleNamespace(calls=calls, torch=torch, transformers=transformers)


def test_native_api_and_pinned_lazy_aligner(fake_libraries):
    adapter = model.NativeQwen(offline=True)
    samples = np.ones(16000, dtype=np.float32)
    assert not any(kind == "aligner_model" for kind, _ in fake_libraries.calls)
    transcript = adapter.transcribe(samples, None, 1024)
    assert transcript == model.Transcript("Hello, world!", "English")
    assert adapter.align(samples, transcript) == [Word("Hello", 0.1, 0.5), Word("world", 0.5, 0.9)]
    adapter.align(samples, transcript)
    loaded = [(kind, options) for kind, options in fake_libraries.calls if kind.endswith("model")]
    assert [kind for kind, _ in loaded] == ["asr_model", "aligner_model"]
    for _, options in loaded:
        assert options["dtype"] == "bf16"
        assert options["device_map"] == "cuda"
        assert options["attn_implementation"] == "sdpa"
    for kind, options in fake_libraries.calls:
        if kind in {"processor", "asr_model", "aligner_model"}:
            expected = (
                model.ALIGNER_REVISION if "ForcedAligner" in options["name"] else model.ASR_REVISION
            )
            assert options["revision"] == expected
            assert options["local_files_only"] is True
            assert options["trust_remote_code"] is False
            assert options["cache_dir"] == str(adapter.local / "models" / "hub")
        if kind == "generate":
            assert options["max_new_tokens"] == 1024
            assert options["do_sample"] is False
            assert options["disable_compile"] is True
        if kind.startswith("prepare_"):
            assert options["processor_kwargs"] == {"sampling_rate": 16000}


def test_forced_language_survives_missing_decoded_language(fake_libraries):
    adapter = model.NativeQwen()
    adapter.processor.decode = lambda *args, **kwargs: {
        "language": None, "transcription": "Привет."
    }
    result = adapter.transcribe(np.ones(16000, dtype=np.float32), "Russian", 1024)
    assert result == model.Transcript("Привет.", "Russian")


def test_generation_at_cap_raises_even_if_last_token_is_eos(fake_libraries):
    adapter = model.NativeQwen()
    with pytest.raises(TranscriptionError, match="may be truncated"):
        adapter.transcribe(np.ones(16000, dtype=np.float32), None, 3)


def test_generation_limit_helper():
    model.check_generation_limit(1023, 1024)
    with pytest.raises(TranscriptionError):
        model.check_generation_limit(1024, 1024)
    with pytest.raises(TranscriptionError):
        model.check_generation_limit(1025, 1024)


def test_unavailable_cuda_does_not_load_model(fake_libraries):
    fake_libraries.torch.cuda.is_available = lambda: False
    with pytest.raises(ModelError, match="CUDA is unavailable"):
        model.NativeQwen()
    assert fake_libraries.calls == []


def test_unsupported_bf16(fake_libraries):
    fake_libraries.torch.cuda.is_bf16_supported = lambda: False
    with pytest.raises(ModelError, match="does not support BF16"):
        model.NativeQwen()


def test_no_visible_device_despite_available_runtime(fake_libraries):
    fake_libraries.torch.cuda.device_count = lambda: 0
    with pytest.raises(ModelError, match="CUDA is unavailable"):
        model.NativeQwen()
    assert fake_libraries.calls == []


@pytest.mark.parametrize("error", [AssertionError("Invalid device id"), RuntimeError("Driver")])
def test_device_query_failures_are_readable(fake_libraries, error):
    def broken():
        raise error

    fake_libraries.torch.cuda.is_bf16_supported = broken
    with pytest.raises(ModelError, match="CUDA initialization failed"):
        model.NativeQwen()


def test_missing_cached_model_explains_offline(fake_libraries):
    def missing(*args, **kwargs):
        raise OSError("not cached")

    fake_libraries.transformers.AutoProcessor.from_pretrained = missing
    with pytest.raises(ModelError, match="without --offline"):
        model.NativeQwen(offline=True)


def test_missing_dependency(fake_libraries, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ModelError, match="Missing model dependency"):
        model.NativeQwen()


def test_inference_failure_is_contextual(fake_libraries):
    adapter = model.NativeQwen()

    def oom(**kwargs):
        raise RuntimeError("CUDA out of memory")

    adapter.model.generate = oom
    with pytest.raises(TranscriptionError, match="ASR inference failed.*out of memory"):
        adapter.transcribe(np.ones(16000, dtype=np.float32), None, 1024)


def test_alignment_rejects_missing_word(fake_libraries):
    adapter = model.NativeQwen()
    adapter._load_aligner()
    adapter.aligner_processor.decode_forced_alignment = lambda **kwargs: [[
        {"text": "Hello", "start_time": 0.1, "end_time": 0.2},
    ]]
    with pytest.raises(TranscriptionError, match="every transcript word"):
        adapter.align(np.ones(16000, dtype=np.float32), model.Transcript("Hello world", "English"))


def test_aligner_language_support_and_mixed_ru_en():
    assert model.alignment_language("Russian") == "Russian"
    assert model.alignment_language("English") == "English"
    assert model.alignment_language("Russian, English") is None
    assert model.alignment_language(None) is None
    with pytest.raises(TranscriptionError, match="does not support"):
        model.alignment_language("Arabic")
