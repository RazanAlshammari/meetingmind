import uuid
import os
import json
import subprocess
import sys
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

meetings_db = {}

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="meeting_chunks")

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".mp4"}

@app.get("/")
def read_root():
    return {"message": "MeetingMind API is running"}

@app.post("/meetings")
async def create_meeting(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    meeting_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{meeting_id}{extension}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    meetings_db[meeting_id] = {
        "meeting_id": meeting_id,
        "original_filename": file.filename,
        "file_path": save_path,
        "status": "uploaded"
    }

    return meetings_db[meeting_id]

@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meetings_db[meeting_id]

def run_pipeline(meeting_id: str, audio_path: str):
    meetings_db[meeting_id]["status"] = "processing"
    try:
        subprocess.run([sys.executable, "transcribe.py", audio_path], check=True)
        subprocess.run([sys.executable, "diarize.py", audio_path], check=True)
        subprocess.run([sys.executable, "merge.py", audio_path], check=True)
        subprocess.run([sys.executable, "analyze.py", audio_path], check=True)
        subprocess.run([sys.executable, "chunk.py", audio_path], check=True)
        subprocess.run([sys.executable, "embed.py", audio_path], check=True)

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        result_path = f"output/{base_name}_analysis.json"
        with open(result_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        meetings_db[meeting_id]["status"] = "done"
        meetings_db[meeting_id]["result"] = analysis
    except subprocess.CalledProcessError as e:
        meetings_db[meeting_id]["status"] = "failed"
        meetings_db[meeting_id]["error"] = str(e)

@app.post("/meetings/{meeting_id}/analyze")
def analyze_meeting(meeting_id: str, background_tasks: BackgroundTasks):
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="Meeting not found")

    audio_path = meetings_db[meeting_id]["file_path"]
    meetings_db[meeting_id]["status"] = "queued"
    background_tasks.add_task(run_pipeline, meeting_id, audio_path)

    return {"meeting_id": meeting_id, "status": "queued"}

@app.post("/ask")
def ask_question(question: str):
    question_embedding = embedding_model.encode([question]).tolist()
    results = chroma_collection.query(query_embeddings=question_embedding, n_results=3)

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(
            f"[Meeting: {meta['meeting']}, Speaker: {meta['speaker']}, {meta['start']:.1f}s-{meta['end']:.1f}s]\n{doc}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = """You are an assistant that answers questions about meetings using ONLY the provided context.
Always cite the meeting, speaker, and timestamp for any claim you make.
If the context doesn't contain the answer, say so clearly instead of guessing."""

    payload = {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/chat", json=payload)
    answer = response.json()["message"]["content"]

    return {"question": question, "answer": answer, "sources": results["metadatas"][0]}