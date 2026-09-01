import json
import requests

# 1. نقرأ نفس الترانسكربت من Phase 2
with open("output/sample_meeting_transcript.json", "r", encoding="utf-8") as f:
    transcript_data = json.load(f)

full_text = " ".join(segment["text"] for segment in transcript_data["segments"])

print("النص الأصلي:")
print(full_text)
print()

# 2. نجهز الطلب لـ Ollama مع system prompt يحدد شكل الـ JSON المطلوب
url = "http://localhost:11434/api/chat"

system_prompt = """You are an assistant that extracts structured information from meeting or speech transcripts.
Always respond with ONLY valid JSON, with no extra text, no explanations, and no markdown code fences.
Use exactly this structure:

{
  "decisions": ["..."],
  "action_items": [{"person": "...", "task": "...", "deadline": "..."}],
  "deadlines": ["..."],
  "open_questions": ["..."]
}

If a category has nothing to report, return an empty list for it. Do not invent information that isn't in the transcript."""

payload = {
    "model": "qwen2.5:7b",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract structured information from this transcript:\n\n{full_text}"}
    ],
    "stream": False
}

# 3. نرسل الطلب
response = requests.post(url, json=payload)
response.raise_for_status()

raw_content = response.json()["message"]["content"].strip()

# 4. كود دفاعي: أحياناً الموديل يحط الـ JSON جوه ```json ... ``` رغم اننا طلبنا منه ما يسوي كذا
if raw_content.startswith("```"):
    raw_content = raw_content.strip("`")
    if raw_content.startswith("json"):
        raw_content = raw_content[4:].strip()

# 5. نحاول نحوله لـ JSON فعلي، ولو فشل نطبع الخطأ بدل ما البرنامج يطيح فجأة
try:
    extracted = json.loads(raw_content)
except json.JSONDecodeError as e:
    print("تحذير: الموديل ما رجع JSON صحيح!")
    print("الرد الخام:")
    print(raw_content)
    raise e

print("النتيجة المستخرجة:")
print(json.dumps(extracted, ensure_ascii=False, indent=2))

# 6. نحفظها بملف
with open("output/sample_meeting_extracted.json", "w", encoding="utf-8") as f:
    json.dump(extracted, f, ensure_ascii=False, indent=2)

print("\nتم حفظ النتيجة بملف: output/sample_meeting_extracted.json")