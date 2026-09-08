from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "We need to migrate the database to PostgreSQL.",
    "Let's switch our database to Postgres.",
    "The weather is nice today."
]

embeddings = model.encode(sentences)

print(f"شكل المتجه الواحد: {embeddings.shape}")

similarity = model.similarity(embeddings, embeddings)
print("\nمصفوفة التشابه بين الجمل:")
print(similarity)