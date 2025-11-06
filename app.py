import os, re, json, streamlit as st, nltk
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# NLTK Fix
# -----------------------------
def ensure_nltk_tokenizer():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except:
            pass

ensure_nltk_tokenizer()

# -----------------------------
# CONFIG
# -----------------------------
GEMINI_API_KEY = "AIzaSyBnNcI3bPJWgdbbTGBqqRr3tM9hSWYvWaQ"  # Replace with your key
PDF_PATH = "NFHS-5_Phase-II_0.pdf"
MODEL = "gemini-2.5-flash-lite"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📘 NFHS PDF Q&A", layout="wide")
st.title("📘 Research PDF Q&A — Multi Chunking")

# -----------------------------
# Gemini Setup
# -----------------------------
def configure_gemini():
    if not GEMINI_API_KEY:
        return False, "❌ Missing API Key"
    genai.configure(api_key=GEMINI_API_KEY)
    return True, ""

def gemini_answer(prompt, max_tokens=500):
    ok, msg = configure_gemini()
    if not ok:
        return msg

    model = genai.GenerativeModel(MODEL, generation_config={"max_output_tokens": max_tokens})
    res = model.generate_content(prompt)
    return res.text

# -----------------------------
# Load PDF
# -----------------------------
def read_pdf(path):
    if not os.path.exists(path):
        return ""
    t = ""
    for p in PdfReader(path).pages:
        content = p.extract_text()
        if content:
            t += content + "\n"
    return t.strip()

doc_text = read_pdf(PDF_PATH)
if not doc_text:
    st.error("❌ PDF not found.")
    st.stop()

# -----------------------------
# Chunk Functions
# -----------------------------
def fixed_chunks(t, size=1000):
    t = t.replace("\n", " ")
    return [t[i:i + size] for i in range(0, len(t), size)]

def recursive_chunks(t, size=1000, ov=200):
    step = size - ov
    return [t[i:i + size] for i in range(0, len(t), step)]

def sentence_chunks(t, n=5):
    s = nltk.sent_tokenize(t)
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

def paragraph_chunks(t, min_len=200):
    p = [x.strip() for x in t.split("\n\n") if x.strip()]
    out = []
    buf = ""
    for x in p:
        if len(x) < min_len:
            buf += " " + x
        else:
            if buf:
                out.append((buf + " " + x).strip())
                buf = ""
            else:
                out.append(x)
    if buf:
        out.append(buf)
    return out

def semantic_chunks(t, win=6, thr=0.65):
    s = nltk.sent_tokenize(t)
    m = SentenceTransformer("all-MiniLM-L6-v2")
    e = m.encode(s, convert_to_numpy=True)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-10)
    out = []
    i = 0
    while i < len(s):
        j = min(i + win, len(s))
        while j < len(s):
            if float(np.dot(e[i:j].mean(axis=0), e[j])) >= thr:
                j += 1
            else:
                break
        out.append(" ".join(s[i:j]))
        i = j
    return out

# ✅ NEW: Overlapping Chunking (replaces Agentic)
def overlapping_chunks(t, size=1000, overlap=300):
    """
    Sliding window chunking with overlap.
    Ensures smoother context continuity.
    """
    t = t.replace("\n", " ")
    step = size - overlap
    chunks = []
    for i in range(0, len(t), step):
        chunk = t[i:i + size]
        chunks.append(chunk)
        if i + size >= len(t):
            break
    return chunks

# -----------------------------
# Retrieve Top-K chunks
# -----------------------------
def retrieve_top(chunks, q, k=5):
    try:
        m = SentenceTransformer("all-MiniLM-L6-v2")
        ce = m.encode(chunks, convert_to_numpy=True)
        qe = m.encode([q], convert_to_numpy=True)[0]
        sims = (ce @ qe) / (np.linalg.norm(ce, axis=1) * np.linalg.norm(qe) + 1e-10)
        idx = np.argsort(sims)[-k:][::-1]
        return [chunks[i] for i in idx]
    except:
        scores = [sum(w.lower() in c.lower() for w in q.split()) for c in chunks]
        idx = np.argsort(scores)[-k:][::-1]
        return [chunks[i] for i in idx]

# -----------------------------
# Prompt
# -----------------------------
def answer_from_chunks(chunks, q):
    ctx = "\n\n".join(chunks)
    prm = f"""
Use ONLY this context to answer.
If unclear, say: "Not clearly answered in the document."

Context:
{ctx}

Question: {q}

Answer in 5-7 bullet points with facts from the document:
"""
    return gemini_answer(prm)

# -----------------------------
# UI
# -----------------------------
method = st.sidebar.selectbox("Method", [
    "Fixed chunking", "Recursive chunking", "Semantic chunking",
    "Sentence chunking", "Paragraph chunking", "Overlapping chunking"
])

q = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not q.strip():
        st.warning("Enter a question")
        st.stop()

    with st.spinner("Thinking..."):

        # ---- Get Chunks ----
        if method == "Fixed chunking":
            ch = fixed_chunks(doc_text)
        elif method == "Recursive chunking":
            ch = recursive_chunks(doc_text)
        elif method == "Semantic chunking":
            ch = semantic_chunks(doc_text)
        elif method == "Sentence chunking":
            ch = sentence_chunks(doc_text)
        elif method == "Paragraph chunking":
            ch = paragraph_chunks(doc_text)
        elif method == "Overlapping chunking":
            ch = overlapping_chunks(doc_text)

        # ---- Retrieve Top-5 chunks ----
        top = retrieve_top(ch, q, k=5)
        ans = answer_from_chunks(top, q)

        st.success(f"Answer ({method})")
        st.write(ans)

        # ✅ Show top-5 chunks on demand
        with st.expander("🔍 Show Top 5 Evidence Chunks"):
            for i, ck in enumerate(top, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(ck)
                st.markdown("---")

st.caption("✅ Click 'Show Evidence' to view the chunks used.")
