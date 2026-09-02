import sys
import os
import json
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# 1. نقرأ اسم ملف الصوت من الترمنال، أو نستخدم jfk.flac كافتراضي
audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/jfk.flac"

# 2. نحمّل المتغيرات من .env (يشمل HUGGINGFACE_TOKEN)
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# 3. نحمّل الموديل المدرب مسبقاً من Hugging Face
print("جاري تحميل موديل الديارايزيشن... (أول مرة بس، بعدها يكون محفوظ محلياً)")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token
)

# 4. نقرأ ملف الصوت عن طريق soundfile
print("جاري تحليل الملف الصوتي...")
audio_array, sample_rate = sf.read(audio_path)

if audio_array.ndim == 1:
    audio_array = audio_array[np.newaxis, :]
else:
    audio_array = audio_array.T

waveform = torch.from_numpy(audio_array).float()
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# 5. نطبع النتيجة ونجمعها بقائمة
speaker_segments = []
for turn, speaker in diarization.speaker_diarization:
    print(f"[{turn.start:.2f}s -> {turn.end:.2f}s] {speaker}")
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

# 6. نشتق اسم ملف الحفظ تلقائياً من اسم ملف الصوت
base_name = os.path.splitext(os.path.basename(audio_path))[0]
output_path = f"output/{base_name}_diarization.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(speaker_segments, f, ensure_ascii=False, indent=2)

print(f"\nتم حفظ نتيجة الديارايزيشن بملف: {output_path}")