import sys
import chromadb
from sentence_transformers import SentenceTransformer
import requests

question = sys.argv[1] if len(sys.argv) > 1 else "What did they talk about?"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="meeting_chunks")

question_embedding = model.encode([question]).tolist()
results = collection.query(query_embeddings=question_embedding, n_results=3)

context_parts = []
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    context_parts.append(
        f"[Meeting: {meta['meeting']}, Speaker: {meta['speaker']}, {meta['start']:.1f}s-{meta['end']:.1f}s]\n{doc}"
    )
context = "\n\n".join(context_parts)

print("القطع المسترجعة (الدليل):\n")
print(context)
print()

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

print("الإجابة:\n")
print(answer)