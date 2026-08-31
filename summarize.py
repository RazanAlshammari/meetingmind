import json
import requests

# 1. نقرأ الترانسكربت اللي حفظناه بمرحلة Phase 2
with open("output/jfk_transcript.json", "r", encoding="utf-8") as f:
    transcript_data = json.load(f)

# ندمج كل المقاطع بنص واحد متواصل (ما نحتاج التوقيتات هنا، بس النص)
full_text = " ".join(segment["text"] for segment in transcript_data["segments"])

print("النص الأصلي:")
print(full_text)
print()

# 2. نجهز الطلب لـ Ollama
url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen2.5:7b",
    "messages": [
        {
            "role": "system",
            "content": "You are an assistant that writes clear, concise summaries of meeting or speech transcripts."
        },
        {
            "role": "user",
            "content": f"Summarize the following transcript in 2-3 sentences:\n\n{full_text}"
        }
    ],
    "stream": False  # عادةً الرد يجي كلمة كلمة (زي ChatGPT)، هنا نطفي هذا ونستنى الرد كامل دفعة وحدة، أبسط للبداية
}

# 3. نرسل الطلب ونستقبل الرد
response = requests.post(url, json=payload)
response.raise_for_status()  # لو صار خطأ بالسيرفر، يطلع لنا رسالة واضحة فورًا بدل ما يكمل بصمت

result = response.json()
summary = result["message"]["content"]

print("الملخص:")
print(summary)

# 4. نحفظ الملخص بملف
with open("output/jfk_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nتم حفظ الملخص بملف: output/jfk_summary.txt")