import uuid
import os
from fastapi import FastAPI, UploadFile, File, HTTPException

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