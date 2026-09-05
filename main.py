import uuid
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import subprocess
import sys
import json
from fastapi import BackgroundTasks

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

meetings_db = {}

@app.get("/")
def read_root():
    return {"message": "MeetingMind API is running"}

@app.post("/meetings")
async def create_meeting(file: UploadFile = File(...)):
    meeting_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
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