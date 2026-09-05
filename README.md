# MeetingMind — Privacy-First AI Meeting Intelligence

MeetingMind processes meeting recordings **entirely on your own machine** —
no audio, transcript, or meeting content is ever sent to an external API
(no OpenAI, no Gemini, no Claude API). It transcribes the meeting, identifies
who spoke, and extracts a summary, decisions, action items, deadlines, and
open questions — all using local AI models.

This is a portfolio AI Engineering project built incrementally, one phase
at a time, with each stage tested on real audio before moving to the next.

## What it does

Given an audio/video file of a meeting, MeetingMind runs it through a local
pipeline:

```
Audio/Video
    -> Speech-to-Text (faster-whisper)
    -> Speaker Diarization (pyannote.audio)
    -> Transcript + Speaker merge
    -> Local LLM extraction (Ollama + Qwen2.5-7B)
    -> Summary, Decisions, Action Items, Deadlines, Open Questions
```

Speakers are identified as anonymous labels (`SPEAKER_00`, `SPEAKER_01`, ...),
not real names — this is a deliberate scope decision, not a limitation of
the pipeline (see "Known limitations" below).

## Tech stack

- **Speech-to-Text**: faster-whisper (CTranslate2-based Whisper), CPU inference
- **Speaker Diarization**: pyannote.audio 4.x (`speaker-diarization-3.1`)
- **Local LLM**: Qwen2.5-7B-Instruct, served via Ollama
- **API**: FastAPI + uvicorn
- **Runs fully on CPU** — no GPU required (tested on an Intel i7 laptop, 16GB RAM)

## Prerequisites

- Python 3.12+
- Ollama installed (https://ollama.com), with the model pulled:
```
  ollama pull qwen2.5:7b
```
- A Hugging Face account with a **read-only** access token, and access
  accepted on the gated pyannote model pages:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`
  - `pyannote/speaker-diarization-community-1`

## Setup

1. Clone the repo and create a virtual environment:
```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```
2. Create a `.env` file in the project root with your Hugging Face token:
```
   HUGGINGFACE_TOKEN=your_token_here
```
3. Make sure Ollama is running in the background (it usually starts
   automatically after installation).

## Running the API

```
uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive docs
(Swagger UI) are at `http://127.0.0.1:8000/docs`.

## API usage

### 1. Upload a meeting recording

```
POST /meetings
```

Accepts a file upload (`.mp3`, `.wav`, `.flac`, `.m4a`, `.mp4` — anything
else is rejected with a 400 error). Returns a `meeting_id`:

```json
{
  "meeting_id": "4402bf90-5298-4e61-85e9-1978e07ba274",
  "original_filename": "meeting.wav",
  "file_path": "uploads\\4402bf90-5298-4e61-85e9-1978e07ba274.wav",
  "status": "uploaded"
}
```

### 2. Trigger analysis

```
POST /meetings/{meeting_id}/analyze
```

Starts the full pipeline (transcription, diarization, merging, LLM
extraction) as a background job and returns immediately:

```json
{ "meeting_id": "4402bf90-...", "status": "queued" }
```

Processing time depends on audio length — expect roughly 1-3 minutes for a
short (~1 minute) clip on CPU.

### 3. Check status / get the result

```
GET /meetings/{meeting_id}
```

Poll this endpoint until `status` is `"done"` (or `"failed"`):

```json
{
  "meeting_id": "4402bf90-...",
  "status": "done",
  "result": {
    "summary": "...",
    "decisions": ["..."],
    "action_items": [
      { "person": "SPEAKER_01", "task": "...", "deadline": "..." }
    ],
    "deadlines": ["..."],
    "open_questions": ["..."]
  }
}
```

## Known limitations (honest, current status)

- Meeting records are stored **in memory only** — restarting the server
  clears all meeting history and results.
- No protection against re-triggering analysis on an already-processed
  meeting (it simply re-runs and overwrites the previous result).
- Diarization has only been validated on a clean two-speaker recording with
  no overlapping speech — not yet tested against messy, realistic
  multi-speaker conversation (planned: evaluation against the AMI Meeting
  Corpus).
- The local LLM occasionally produces internally inconsistent output
  between runs on identical input (a known, observed characteristic of
  local LLM sampling, not a code bug) — structured fields are more
  reliable than free-text summary prose.
- No authentication or multi-user support — this is a single-user local
  tool, not a hosted multi-tenant service.

## Roadmap

- [x] V1: local pipeline (transcription, diarization, structured extraction) + REST API
- [ ] V2: "Ask Your Meetings" — RAG-based Q&A across multiple meetings, with evidence and timestamps
- [ ] Docker packaging
- [ ] Arabic language support (V3), then mixed Arabic/English (V4)