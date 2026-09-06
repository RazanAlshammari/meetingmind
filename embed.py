import sys
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/jfk.flac"
base_name = os.path.splitext(os.path.basename(audio_path))[0]

input_path = f"output/{base_name}_chunks.json"

with open(input_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("جاري تحميل موديل الـ embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="meeting_chunks")

ids = [f"{base_name}_{i}" for i in range(len(chunks))]
documents = [chunk["text"] for chunk in chunks]
embeddings = model.encode(documents).tolist()
metadatas = [
    {
        "meeting": base_name,
        "speaker": chunk["speaker"],
        "start": chunk["start"],
        "end": chunk["end"]
    }
    for chunk in chunks
]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print(f"تم إضافة {len(chunks)} قطعة لقاعدة البيانات المتجهة.")
print(f"إجمالي عدد القطع بالقاعدة الآن: {collection.count()}")