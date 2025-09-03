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
      "Remove any jewelry or tight clothing near the burn (not stuck to the skin).",
      "Cover loosely with cling film or a clean non-fluffy cloth.",
      "Do not use ice, creams, or butter."
    ],
    "when_to_seek_help": "Seek emergency help if the burn is deep, larger than the person's hand, or on the face, hands, feet, or genitals."
  },
  {
    "condition": "bleeding",
    "keywords": ["cut", "bleed", "wound", "hemorrhage"],
    "steps": [
      "Apply firm, direct pressure to the wound with a clean cloth or bandage.",
      "If possible, elevate the injured area above heart level.",
      "If blood soaks through, apply another dressing on top (do not remove the first).",
      "Keep pressure until bleeding stops or help arrives."
    ],
    "when_to_seek_help": "Call emergency services if bleeding is severe, spurting, or does not stop after 10 minutes of pressure."
  },
  {
    "condition": "choking",
    "keywords": ["choke", "airway", "obstruction", "coughing"],
    "steps": [
      "Encourage the person to cough if they can.",
      "If they cannot breathe, perform up to 5 back blows.",
      "If unsuccessful, perform up to 5 abdominal thrusts (Heimlich maneuver).",
      "Repeat until object is expelled or medical help arrives."
    ],
    "when_to_seek_help": "Call emergency services immediately if the person becomes unconscious or cannot breathe at all."
  },
  {
    "condition": "fractures",
    "keywords": ["broken bone", "fracture", "crack", "bone injury"],
    "steps": [
      "Support the injured area with a sling, padding, or bandage.",
      "Keep the person still and calm.",
      "Apply a cold pack wrapped in cloth to reduce swelling.",
      "Do not try to straighten the bone."
    ],
    "when_to_seek_help": "Always seek medical help for suspected fractures."
  },
  {
    "condition": "shock",
    "keywords": ["shock", "faint", "collapse", "low blood pressure"],
    "steps": [
      "Help the person lie down and raise their legs if possible.",
      "Keep them warm with a blanket or coat.",
      "Do not give food or drink.",
      "Check for responsiveness and breathing."
    ],
    "when_to_seek_help": "Call emergency services immediately."
  },
  {
    "condition": "asthma attack",
    "keywords": ["asthma", "wheezing", "difficulty breathing"],
    "steps": [
      "Help them sit upright comfortably.",
      "Reassure them and encourage slow breaths.",
      "Assist them in using their inhaler (usually a reliever inhaler).",
      "Call emergency services if symptoms don’t improve after inhaler use."
    ],
    "when_to_seek_help": "Seek immediate help if breathing worsens or inhaler does not provide relief."
  },
  {
    "condition": "heart attack",
    "keywords": ["chest pain", "heart attack", "myocardial infarction"],
    "steps": [
      "Help the person sit down and keep calm.",
      "Call emergency services immediately.",
      "Give them 300mg aspirin to chew slowly if they are not allergic.",
      "Monitor breathing and prepare for CPR if needed."
    ],
    "when_to_seek_help": "Always call emergency services immediately."
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
