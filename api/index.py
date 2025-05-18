# File: api/index.py
import os
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

app = FastAPI()

# === Config ===
PREPROCESSED_JSON = "cleaned_output.json"
POST_BANK_JSON = "post_bank.json"
EMBEDDING_NPY = "post_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10

encoder = SentenceTransformer(MODEL_NAME)

# === Load data ===
if os.path.exists(POST_BANK_JSON) and os.path.exists(EMBEDDING_NPY):
    with open(POST_BANK_JSON, "r", encoding="utf-8") as f:
        post_bank = json.load(f)
    embeddings = np.load(EMBEDDING_NPY)
else:
    with open(PREPROCESSED_JSON, "r", encoding="utf-8") as f:
        threads = json.load(f)

    post_bank = []
    all_texts = []

    for thread in threads:
        for post in thread["posts"]:
            clean_text = post.get("clean_text", "").strip()
            if clean_text:
                post_bank.append({
                    "text": clean_text,
                    "url": post["post_url"],
                    "thread_slug": thread["topic_slug"]
                })
                all_texts.append(clean_text)

    embeddings = encoder.encode(all_texts, convert_to_numpy=True)
    np.save(EMBEDDING_NPY, embeddings)
    with open(POST_BANK_JSON, "w", encoding="utf-8") as f:
        json.dump(post_bank, f, indent=2)

def semantic_search(query, top_k=TOP_K):
    query_vec = encoder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [{
        "response": post_bank[i]["text"],
        "reference_link": post_bank[i]["url"],
        "score": float(sims[i])
    } for i in top_indices]

def load_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise Exception("Environment variable 'GROQ_API_KEY' not set.")
    return key

def query_groq_with_context(query, context_responses, groq_api_key):
    context_prompt = "\n\n".join([f"- {r['response']}" for r in context_responses])
    system_prompt = (
        "You are an assistant helping IITM Degree learners. "
        "Use the context below to answer the student's query accurately and don't assume things up.\n\n"
        f"Context:\n{context_prompt}\n\nQuery: {query}\nAnswer:"
    )

    openai.api_key = groq_api_key
    openai.api_base = "https://api.groq.com/openai/v1"

    response = openai.ChatCompletion.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful online TA."},
            {"role": "user", "content": system_prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )

    return response["choices"][0]["message"]["content"]

class QueryRequest(BaseModel):
    query: str

@app.post("/")
def handle_query(request_data: QueryRequest):
    query = request_data.query.strip()
    if not query:
        return {"error": "Empty query."}

    top_matches = semantic_search(query)
    groq_api_key = load_groq_api_key()
    llm_answer = query_groq_with_context(query, top_matches, groq_api_key)

    return {
        "query": query,
        "llm_answer": llm_answer,
        "top_references": top_matches
    }
