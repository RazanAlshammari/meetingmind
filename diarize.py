import os
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import json

# 1. نحمّل المتغيرات من .env (يشمل HUGGINGFACE_TOKEN)
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# 2. نحمّل الموديل المدرب مسبقاً من Hugging Face
print("جاري تحميل موديل الديارايزيشن... (أول مرة بس، بعدها يكون محفوظ محلياً)")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token
)

# 3. نقرأ ملف الصوت عن طريق soundfile (بديل مستقل عن FFmpeg/torchcodec كلياً)
print("جاري تحليل الملف الصوتي...")
audio_array, sample_rate = sf.read("data/jfk.flac")

# soundfile يرجع الصوت بشكل (samples,) لو صوت واحد (mono)، أو (samples, channels) لو أكثر
# بينما pyannote يحتاج شكل (channels, samples) كـ torch tensor
if audio_array.ndim == 1:
    audio_array = audio_array[np.newaxis, :]
else:
    audio_array = audio_array.T

waveform = torch.from_numpy(audio_array).float()
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# 4. نطبع النتيجة: كل مقطع، وين بدأ ووين خلص، ومين المتحدث

speaker_segments = []
for turn, speaker in diarization.speaker_diarization:
    print(f"[{turn.start:.2f}s -> {turn.end:.2f}s] {speaker}")
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

with open("output/jfk_diarization.json", "w", encoding="utf-8") as f:
    json.dump(speaker_segments, f, ensure_ascii=False, indent=2)

print("\nتم حفظ نتيجة الديارايزيشن بملف: output/jfk_diarization.json")