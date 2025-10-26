"""FastAPI application exposing a Faster Whisper transcription endpoint."""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# Models documented in the faster-whisper README
AVAILABLE_MODELS: Dict[str, str] = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large": "large",
    "distil-small.en": "distil-small.en",
    "distil-medium.en": "distil-medium.en",
    "distil-large-v2": "distil-large-v2",
}

DEFAULT_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
    raise RuntimeError(
        f"Unsupported default model '{DEFAULT_MODEL_NAME}'. "
        f"Update WHISPER_MODEL to one of: {', '.join(sorted(AVAILABLE_MODELS))}."
    )


@lru_cache(maxsize=4)
def load_model(model_name: str) -> WhisperModel:
    """Load and memoise whisper models to avoid repeated downloads."""
    if model_name not in AVAILABLE_MODELS:
        raise KeyError(
            f"Model '{model_name}' is not recognised. Choose from: {', '.join(sorted(AVAILABLE_MODELS))}."
        )

    return WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)


app = FastAPI(title="Faster Whisper API")


@app.on_event("startup")
def preload_default_model() -> None:
    """Warm up the default model when the service starts."""
    load_model(DEFAULT_MODEL_NAME)


def _transcribe(
    file_path: str,
    model_name: str,
    language: Optional[str],
    task: Literal["transcribe", "translate"],
    beam_size: int,
) -> Dict[str, object]:
    model = load_model(model_name)

    segments, info = model.transcribe(
        file_path,
        language=language,
        task=task,
        beam_size=beam_size,
    )

    text = "".join(segment.text for segment in segments)
    segments_payload: List[Dict[str, object]] = [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        for segment in segments
    ]

    return {
        "model": model_name,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "transcription": text.strip(),
        "segments": segments_payload,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model_name: str = Form(DEFAULT_MODEL_NAME),
    language: Optional[str] = Form(None, description="Optional language code override"),
    task: Literal["transcribe", "translate"] = Form("transcribe"),
    beam_size: int = Form(5),
) -> JSONResponse:
    """Transcribe the provided audio file using the selected Faster Whisper model."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model '{model_name}'. Choose from: {', '.join(sorted(AVAILABLE_MODELS))}.",
            },
        )

    try:
        contents = await file.read()
    except Exception as exc:  # pragma: no cover - FastAPI handles request lifecycle
        raise HTTPException(status_code=500, detail={"error": "file_read_error", "message": str(exc)}) from exc

    if not contents:
        raise HTTPException(status_code=400, detail={"error": "empty_file", "message": "Uploaded file is empty."})

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        payload = _transcribe(tmp_path, model_name, language, task, beam_size)
    except Exception as exc:  # pragma: no cover - delegated to model
        raise HTTPException(status_code=500, detail={"error": "transcription_failed", "message": str(exc)}) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    return JSONResponse(content=payload)


@app.get("/models")
def list_models() -> Dict[str, List[str]]:
    """Return the list of available models."""
    return {"available_models": sorted(AVAILABLE_MODELS)}
