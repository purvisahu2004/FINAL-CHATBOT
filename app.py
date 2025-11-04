# app.py
import os
import re
import json
import streamlit as st
import nltk
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# USER CONFIG - EDIT THESE
# -----------------------
# Put your Gemini API key here (NO UI prompt)
GEMINI_API_KEY = "AIzaSyBnNcI3bPJWgdbbTGBqqRr3tM9hSWYvWaQ"   # <<-- REPLACE with your key

# Path to PDF file to use (relative to this script or absolute)
PDF_PATH = "NFHS-5_Phase-II_0.pdf"          # <<-- REPLACE with your PDF file name/path

# Gemini model to use
GEMINI_MODEL = "gemini-2.5-flash-lite"

# -----------------------
# Setup
# -----------------------
nltk.download("punkt", quiet=True)
st.set_page_config(page_title="Researcher PDF Q/A — Multi-Chunking", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: white; color: #0f172a; }
    .block-container { padding: 1.2rem 1.6rem; }
    h1, h2, h3 { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚  Q/A — Multi Chunking")


# -----------------------
# Helpers
# -----------------------
def configure_gemini():
    """Configure google.generativeai with the API key (returns tuple ok,msg)."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("PUT_"):
        return False, "Set GEMINI_API_KEY variable at top of app.py"
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True, ""
    except Exception as e:
        return False, f"genai.configure() failed: {e}"

def safe_genai_generate(prompt_or_inputs, model_name=GEMINI_MODEL, max_output_tokens: int = 512):
    """Safe wrapper around google.generativeai generate_content. Returns text or error message."""
    ok, msg = configure_gemini()
    if not ok:
        return f"[Gemini not configured] {msg}"
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"[Gemini init error] {e}"
    try:
        resp = model.generate_content(prompt_or_inputs, max_output_tokens=max_output_tokens)
        # older sdk uses .text; some return different attributes
        text = getattr(resp, "text", None) or getattr(resp, "output_text", None) or str(resp)
        return text
    except Exception as e:
        return f"[Gemini call failed] {e}"

def read_pdf_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                text += txt + "\n\n"
        return text.strip()
    except Exception as e:
        return ""

# ---------- chunking utilities ----------
def fixed_chunks(text: str, chars: int = 1000):
    t = text.replace("\n", " ")
    return [t[i:i+chars].strip() for i in range(0, len(t), chars) if t[i:i+chars].strip()]

def recursive_chunks(text: str, chunk_size: int = 1000, overlap: int = 150):
    chunks = []
    if not text:
        return chunks
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += step
        if end >= len(text):
            break
    return [c for c in chunks if c]

def sentence_chunks(text: str, sentences_per_chunk: int = 8):
    sents = nltk.sent_tokenize(text)
    chunks = [" ".join(sents[i:i+sentences_per_chunk]) for i in range(0, len(sents), sentences_per_chunk)]
    return [c for c in chunks if c.strip()]

def paragraph_chunks(text: str, min_len: int = 200):
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged = []
    buffer = ""
    for p in parts:
        if len(p) < min_len:
            buffer = (buffer + " " + p).strip()
        else:
            if buffer:
                merged.append((buffer + " " + p).strip())
                buffer = ""
            else:
                merged.append(p)
    if buffer:
        merged.append(buffer)
    return merged

# Semantic chunking method #2: embedding-window (adjacent sentences grouping)
def semantic_chunks_embedding_window(text: str, model_name: str = "all-MiniLM-L6-v2", window_sentences: int = 6, sim_threshold: float = 0.64):
    sents = nltk.sent_tokenize(text)
    if not sents:
        return []
    model = SentenceTransformer(model_name)
    sent_embs = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    # normalize
    norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-10
    emb_norm = sent_embs / norms
    chunks = []
    i = 0
    n = len(sents)
    while i < n:
        j = min(i + window_sentences, n)
        while j < n:
            window_mean = emb_norm[i:j].mean(axis=0)
            next_sent = emb_norm[j]
            sim = float(np.dot(window_mean, next_sent))
            # expand window when similarity high
            if sim >= sim_threshold and (j - i) < (window_sentences * 3):
                j += 1
            else:
                break
        chunk_text = " ".join(sents[i:j])
        chunks.append(chunk_text)
        i = j
    # merge tiny chunks
    merged = []
    for c in chunks:
        if len(c) < 50 and merged:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return merged

# lexical retrieve best chunk (fallback)
def retrieve_best_chunk_lexical(chunks, question):
    if not chunks:
        return ""
    scores = [sum(w.lower() in c.lower() for w in question.split()) for c in chunks]
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]

# Agentic: LLM picks candidate indices then LLM answers using chosen chunks (no chunks shown)
def agentic_select_and_answer(text: str, question: str, candidates_fn, top_k_candidates: int = 12, selection_max: int = 6):
    candidates = candidates_fn(text)
    if not candidates:
        return "No candidate chunks generated from the document."
    # filter candidates by lexical overlap for a compact sample
    scores = [sum(w.lower() in c.lower() for w in question.split()) for c in candidates]
    idxs = np.argsort(scores)[-top_k_candidates:][::-1] if len(candidates) > 0 else []
    candidate_sample = [candidates[i] for i in idxs] if len(idxs) > 0 else candidates[:top_k_candidates]
    if not candidate_sample:
        candidate_sample = candidates[:min(len(candidates), top_k_candidates)]
    # Form numbered list
    numbered = "\n".join([f"{i+1}. {candidate_sample[i][:800].replace('\\n',' ')}" for i in range(len(candidate_sample))])
    sel_prompt = f"""You are a selection assistant. The user question is below. From the numbered candidate chunks, return a JSON array of integers with the most relevant chunk indices (1-based), best first. Return ONLY the JSON array, nothing else.

Question:
{question}

Candidates:
{numbered}

Return e.g. [1,3]"""
    sel_resp = safe_genai_generate(sel_prompt, max_output_tokens=256)
    # parse JSON array from response
    chosen = []
    try:
        m = re.search(r"(\[.*?\])", sel_resp, flags=re.S)
        if m:
            arr = json.loads(m.group(1))
            for idx in arr[:selection_max]:
                if isinstance(idx, int) and 1 <= idx <= len(candidate_sample):
                    chosen.append(candidate_sample[idx-1])
    except Exception:
        chosen = []
    if not chosen:
        chosen = [candidate_sample[0]]
    context = "\n\n".join(chosen)
    ans_prompt = f"""You are an assistant. Use ONLY the provided context to answer the question. If the answer is not present, say "I don't know from the document."

Context:
{context}

Question:
{question}

Answer succinctly:"""
    answer = safe_genai_generate(ans_prompt, max_output_tokens=512)
    return answer

# -----------------------
# Load document once (no UI uploader)
# -----------------------
doc_text = read_pdf_text(PDF_PATH)
if not doc_text:
    st.error(f"Could not load text from PDF at path: {PDF_PATH}\nPlease place the PDF at that path and restart the app.")
    st.stop()

# -----------------------
# Sidebar / method selection & inputs
# -----------------------
st.sidebar.header("Method")
method = st.sidebar.selectbox("Choose method", [
    "Fixed-size chunking",
    "Recursive chunking",
    "Semantic (embedding-window)",
    "Sentence-based chunking",
    "Paragraph-based chunking",
    "Agentic chunking (LLM selects chunks)"
])
st.sidebar.markdown("---")
st.sidebar.write(f"PDF loaded from: `{PDF_PATH}`")
st.sidebar.write("Gemini model: " + GEMINI_MODEL)

# single question input (applies to all methods)
st.markdown("### Enter a single question (applies to currently selected method)")
question = st.text_input("Question")

# run button
if st.button("Run method"):
    if not question.strip():
        st.warning("Please type a question before running.")
        st.stop()
    with st.spinner("Running selected method..."):
        # Fixed
        if method == "Fixed-size chunking":
            chunks = fixed_chunks(doc_text, chars=1000)
            best = retrieve_best_chunk_lexical(chunks, question)
            prompt = f"Answer using only the context below. If not present, say 'I don't know from the document.'\n\nContext:\n{best}\n\nQuestion: {question}"
            out = safe_genai_generate(prompt)
            st.subheader("Answer — Fixed-size chunking")
            st.write(out)

        # Recursive
        elif method == "Recursive chunking":
            chunks = recursive_chunks(doc_text, chunk_size=1000, overlap=150)
            # try embedding search
            try:
                embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                emb_chunks = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
                q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
                sims = (emb_chunks @ q_emb) / (np.linalg.norm(emb_chunks, axis=1) * (np.linalg.norm(q_emb) + 1e-10) + 1e-10)
                best_idx = int(np.argmax(sims))
                context = chunks[best_idx]
            except Exception:
                context = retrieve_best_chunk_lexical(chunks, question)
            prompt = f"Answer using only the context below. If not present, say 'I don't know from the document.'\n\nContext:\n{context}\n\nQuestion: {question}"
            out = safe_genai_generate(prompt)
            st.subheader("Answer — Recursive chunking")
            st.write(out)

        # Semantic
        elif method == "Semantic (embedding-window)":
            try:
                chunks = semantic_chunks_embedding_window(doc_text, model_name="all-MiniLM-L6-v2",
                                                          window_sentences=6, sim_threshold=0.64)
                if not chunks:
                    st.warning("Semantic chunks empty; falling back to fixed chunks.")
                    chunks = fixed_chunks(doc_text, chars=1000)
                embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                chunk_embs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
                q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
                sims = (chunk_embs @ q_emb) / (np.linalg.norm(chunk_embs, axis=1) * (np.linalg.norm(q_emb) + 1e-10) + 1e-10)
                best_idx = int(np.argmax(sims))
                context = chunks[best_idx]
                prompt = f"Answer using only the context below. If not present, say 'I don't know from the document.'\n\nContext:\n{context}\n\nQuestion: {question}"
                out = safe_genai_generate(prompt)
                st.subheader("Answer — Semantic chunking (embedding-window)")
                st.write(out)
            except Exception as e:
                st.error(f"Semantic chunking error: {e}")

        # Sentence-based
        elif method == "Sentence-based chunking":
            chunks = sentence_chunks(doc_text, sentences_per_chunk=8)
            best = retrieve_best_chunk_lexical(chunks, question)
            prompt = f"Answer using only the context below. If not present, say 'I don't know from the document.'\n\nContext:\n{best}\n\nQuestion: {question}"
            out = safe_genai_generate(prompt)
            st.subheader("Answer — Sentence-based chunking")
            st.write(out)

        # Paragraph-based
        elif method == "Paragraph-based chunking":
            chunks = paragraph_chunks(doc_text, min_len=200)
            best = retrieve_best_chunk_lexical(chunks, question)
            prompt = f"Answer using only the context below. If not present, say 'I don't know from the document.'\n\nContext:\n{best}\n\nQuestion: {question}"
            out = safe_genai_generate(prompt)
            st.subheader("Answer — Paragraph-based chunking")
            st.write(out)

        # Agentic chunking
        elif method == "Agentic chunking (LLM selects chunks)":
            def candidates_fn(t):
                pars = paragraph_chunks(t, min_len=120)
                if len(pars) < 6:
                    pars = sentence_chunks(t, sentences_per_chunk=6)
                if len(pars) < 1:
                    pars = fixed_chunks(t, chars=800)
                return pars
            out = agentic_select_and_answer(doc_text, question, candidates_fn, top_k_candidates=12, selection_max=5)
            st.subheader("Answer — Agentic chunking")
            st.write(out)

# footer
st.markdown("---")
st.caption("Notes: Set GEMINI_API_KEY and PDF_PATH at top of app.py. Agentic chunking hides chunks; Gemini picks and answers using selected chunks.")
