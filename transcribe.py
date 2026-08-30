from faster_whisper import WhisperModel

# نحمّل نموذج "small"، ونشغله على المعالج (CPU) بصيغة مضغوطة (int8)
# مناسبة لجهاز بدون كرت شاشة مخصص — هذا هو الاختيار اللي اتفقنا عليه بالستاك
model = WhisperModel("small", device="cpu", compute_type="int8")

# نفرّغ الملف الصوتي
segments, info = model.transcribe("data/jfk.flac")

print(f"اللغة المكتشفة: {info.language} (نسبة الثقة: {info.language_probability:.2f})")

# segments ما هو قائمة جاهزة، هو "مولّد" (generator) — يعني ينتج كل مقطع
# وقت ما نمر عليه بالحلقة for، مو كلهم دفعة وحدة
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")