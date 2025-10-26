# Faster Whisper Transcription API

A minimal FastAPI service exposing [faster-whisper](https://pypi.org/project/faster-whisper/) transcription via an HTTP endpoint. The API is designed to run locally with GPU acceleration (CUDA) and supports switching between all models published in the Faster Whisper README.

## Requirements

* Python 3.9+
* CUDA-capable GPU with the appropriate NVIDIA drivers and CUDA toolkit installed

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note**
> Installing `faster-whisper` will automatically pull the compatible PyTorch + CUDA wheels. Ensure your environment satisfies the GPU requirements described in the [upstream project documentation](https://github.com/SYSTRAN/faster-whisper#installation).

## Configuration

Environment variables can be used to tweak the runtime:

| Variable | Default | Description |
| --- | --- | --- |
| `WHISPER_MODEL` | `medium` | Default model to preload on startup. Must be one of the values returned by `GET /models`. |
| `WHISPER_DEVICE` | `cuda` | Device passed to `WhisperModel`. Set to `cpu` for debugging without a GPU. |
| `WHISPER_COMPUTE_TYPE` | `float16` | Compute type forwarded to `WhisperModel` (e.g. `float16`, `int8_float16`). |

## Running the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

After startup the service preloads the default Whisper model on the GPU.

## Endpoints

### `GET /models`

Returns the list of model identifiers accepted by the API.

```json
{
  "available_models": ["base", "base.en", "distil-large-v2", "distil-medium.en", "distil-small.en", "large", "large-v1", "large-v2", "large-v3", "medium", "medium.en", "small", "small.en", "tiny", "tiny.en"]
}
```

### `POST /transcribe`

Multipart endpoint that accepts an audio file along with optional parameters to control the transcription.

Form fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | File | – | Audio file to transcribe (required). |
| `model_name` | String | `WHISPER_MODEL` | Whisper model to use. |
| `language` | String | `null` | Optional language code override. Leave empty to auto-detect. |
| `task` | String | `transcribe` | Either `transcribe` or `translate`. |
| `beam_size` | Integer | `5` | Beam size to use during decoding. |

Example request with `curl`:

```bash
curl -X POST \
     -F "file=@/path/to/audio.wav" \
     -F "model_name=medium" \
     http://localhost:8000/transcribe
```

Example response:

```json
{
  "model": "medium",
  "language": "en",
  "language_probability": 0.99,
  "duration": 12.34,
  "transcription": "Hello world!",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 4.2,
      "text": " Hello",
      "tokens": [50364, 2425, 50517],
      "temperature": 0.0,
      "avg_logprob": -0.12,
      "compression_ratio": 1.5,
      "no_speech_prob": 0.02
    }
  ]
}
```

## Development

The API automatically caches loaded models to avoid redundant downloads. Adjust the maximum cache size in `app/main.py` if you plan to serve many models concurrently.

