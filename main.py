from fastapi import FastAPI, Form
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# ---------------- Knowledge Base ---------------- #
first_aid_data = [
    {
        "condition": "burns",
        "keywords": ["burn", "fire injury", "scald", "flame"],
        "steps": [
            "Cool the burn under cool running water for at least 20 minutes.",
            "Remove jewelry or tight clothing near the burn.",
            "Cover loosely with cling film or a clean cloth.",
            "Do not use ice, creams, or butter."
        ],
        "when_to_seek_help": "Seek emergency help if the burn is deep, larger than the hand, or on face, hands, feet, or genitals."
    },
    {
        "condition": "bleeding",
        "keywords": ["cut", "bleed", "wound", "hemorrhage"],
        "steps": [
            "Apply firm, direct pressure with a clean cloth.",
            "Elevate the injured area above heart level if possible.",
            "If blood soaks through, add another dressing on top.",
            "Do not remove the first dressing."
        ],
        "when_to_seek_help": "Call emergency services if bleeding is severe or does not stop after 10 minutes."
    }
]

# ---------------- Embeddings ---------------- #
model = SentenceTransformer('all-MiniLM-L6-v2')
conditions = [item["condition"] for item in first_aid_data]
embeddings = model.encode(conditions, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def get_first_aid_response(query):
    query = query.lower()
    
    # Keyword search first
    for item in first_aid_data:
        if any(kw in query for kw in item["keywords"]):
            return item
    
    # Embedding fallback
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=1)
    return first_aid_data[I[0][0]]

# ---------------- API Endpoint ---------------- #
@app.post("/ask")
async def ask(query: str = Form(...)):
    response = get_first_aid_response(query)
    return {
        "condition": response["condition"],
        "steps": response["steps"],
        "when_to_seek_help": response["when_to_seek_help"]
    }
