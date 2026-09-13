# MeetingMind

A privacy-first AI meeting intelligence system that runs entirely on local hardware — no audio, transcripts, or meeting content ever leave your machine. Built as a portfolio project to demonstrate a full local AI pipeline: speech-to-text, speaker diarization, structured LLM extraction, and retrieval-augmented generation (RAG) over meeting history.

## Why local-only

Meeting recordings are sensitive. MeetingMind is built around a simple constraint: every model — speech-to-text, diarization, the LLM, and the embedding model used for search — runs on your machine via [Ollama](https://ollama.com) and local Python libraries. Nothing is uploaded to a third-party API. This shapes every technical decision in the project, including which models were chosen (small enough to run acceptably on a CPU-only laptop) and how the API server is configured (binds to `127.0.0.1` by default).

## What it does

**V1 — Meeting Intelligence.** Upload a recording of a meeting; MeetingMind transcribes it, identifies who spoke when, and extracts a structured summary: key decisions, action items (with owner and deadline where mentioned), deadlines, and open questions.

**V2 — Ask Your Meetings.** Once a meeting is processed, ask natural-language questions about it (or across all processed meetings) and get an answer with citations — which meeting, which speaker, and the exact timestamp the answer came from. If the answer isn't in your meetings, it says so instead of guessing.

## Pipeline

```
Audio/Video file
      │
      ▼
faster-whisper (speech-to-text)
      │
      ▼
pyannote.audio (speaker diarization)
      │
      ▼
custom merge (aligns transcript with speaker turns)
      │
      ▼
Qwen2.5-7B via Ollama (structured extraction: summary, decisions,
                        action items, deadlines, open questions)
      │
      ▼
chunking (one chunk per speaker turn) → all-MiniLM-L6-v2 embeddings
      │
      ▼
Chroma (local vector database)
      │
      ▼
POST /ask → embed question → retrieve top-3 relevant chunks →
            Qwen2.5-7B answers with citations
```

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI + uvicorn, async background jobs |
| Speech-to-text | faster-whisper (small/medium, int8) |
| Speaker diarization | pyannote.audio 4.0.7 |
| Local LLM | Qwen2.5-7B-Instruct via Ollama |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector search | Chroma (persistent, local) |

## API

- `POST /meetings` — upload an audio/video file (`.mp3`, `.wav`, `.flac`, `.m4a`, `.mp4`)
- `POST /meetings/{id}/analyze` — run the full pipeline on an uploaded meeting
- `GET /meetings/{id}` — check status (`uploaded` → `queued` → `processing` → `done`/`failed`) and get results
- `POST /ask` — ask a question, optionally scoped to one meeting via `meeting_id`, or across all processed meetings if omitted

Full interactive docs at `/docs` once the server is running.

## Running it locally

```bash
git clone https://github.com/RazanAlshammari/meetingmind.git
cd meetingmind
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

You'll also need [Ollama](https://ollama.com) installed and running, with the model pulled:
```bash
ollama pull qwen2.5:7b
```

Then start the API:
```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` to try it.

## Running it in Docker

A `Dockerfile` is included that containerizes the API service (code + Python dependencies). It's written to connect to Ollama running on the host machine via `host.docker.internal`, rather than containerizing the full local AI stack (Ollama's own container setup and large model downloads add real complexity without changing what the project demonstrates).

**Honest note:** this `Dockerfile` has been written and reviewed but not build-tested end-to-end — the development machine used for this project has CPU virtualization disabled in BIOS, which Docker Desktop requires. The Dockerfile follows standard, well-established patterns (slim Python base image, dependency layer caching, exposed port, `uvicorn` entrypoint) and is expected to build correctly on a machine with virtualization enabled; it just hasn't been verified with an actual `docker build` run in this environment yet.

## Evaluated on real meetings

Beyond hand-written test clips, the full pipeline (V1 analysis + V2 chunking/embedding/retrieval) was run end-to-end on two real recordings from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) — unscripted, multi-speaker, 19–21 minute product-design meetings. Both produced correctly structured analysis and answered follow-up questions with accurate citations, including one case where the system correctly declined to overclaim ("the transcript mentions d-pads and buttons, not an explicit joystick") rather than guessing.

## Known limitations

Documented honestly rather than glossed over — these are real trade-offs made along the way, not oversights discovered later:

- **In-memory meeting registry.** Meeting metadata and status live in memory and reset on server restart; the vector database (Chroma) is already persistent, and moving meeting metadata alongside it is a natural next step.
- **No concurrency guard on `/analyze`.** Triggering analysis twice for the same meeting in quick succession causes a race condition (observed and reproduced during testing). Low-probability in normal use; not yet hardened against.
- **Chunking has no size cap.** Chunks are one per speaker turn, which keeps attribution clean but means a very long uninterrupted turn could exceed the embedding model's ~256-token window during vector search (the full text is still stored and still reaches the LLM once retrieved — this affects search ranking quality, not data loss). Not yet observed to be a problem on real meetings tested so far.
- **Chroma + long-running server processes.** If chunks are added to the vector database by a separate process while the API server is already running, the server's connection can go stale until restarted. Understood and documented; not yet auto-recovered.

## Roadmap

V3 will add Arabic support; V4, mixed Arabic/English meetings. Both are future work, not started.