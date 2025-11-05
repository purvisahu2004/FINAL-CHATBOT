# app.py
# app.py
import os, re, json, streamlit as st, nltk
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
# NLTK sentence tokenizer fix for Streamlit Cloud


def ensure_nltk_tokenizer():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Newer NLTK versions also require punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except:
            pass  # some NLTK versions don't have punkt_tab

ensure_nltk_tokenizer()

# -----------------------
# USER CONFIG
# -----------------------
GEMINI_API_KEY = "AIzaSyBnNcI3bPJWgdbbTGBqqRr3tM9hSWYvWaQ"   # ✅ Add your Gemini key here
PDF_PATH = "NFHS-5_Phase-II_0.pdf"     # ✅ Your PDF file path
GEMINI_MODEL = "gemini-2.5-flash-lite"

# -----------------------
# Streamlit UI
# -----------------------
nltk.download("punkt", quiet=True)
st.set_page_config(page_title="📘 Multi-Chunk PDF Q&A", layout="wide")

st.markdown("""
<style>
.stApp { background:white; color:black; }
h1,h2,h3,label { color:black !important; }
.stTextInput>div>div>input { color:white; }
</style>
""", unsafe_allow_html=True)

st.title("📘 Research PDF Q&A — Multi Chunking")

# -----------------------
# Gemini Setup
# -----------------------
def configure_gemini():
    if not GEMINI_API_KEY:
        return False, "❌ Add your Gemini API key in code"
    genai.configure(api_key=GEMINI_API_KEY)
    return True, ""

def gemini_answer(prompt, max_tokens=500):
    ok, msg = configure_gemini()
    if not ok: return msg

    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"max_output_tokens": max_tokens}
    )
    res = model.generate_content(prompt)
    return res.text

# -----------------------
# PDF Load
# -----------------------
def read_pdf(path):
    if not os.path.exists(path): return ""
    try:
        text = ""
        for p in PdfReader(path).pages:
            t = p.extract_text()
            if t: text += t + "\n"
        return text.strip()
    except:
        return ""

doc_text = read_pdf(PDF_PATH)
if not doc_text:
    st.error(f"❌ PDF not found at `{PDF_PATH}`. Put file and restart.")
    st.stop()

# -----------------------
# Chunkers
# -----------------------
def fixed_chunks(t, size=2000):
    t=t.replace("\n"," "); 
    return [t[i:i+size] for i in range(0,len(t),size)]

def recursive_chunks(t, size=2000, overlap=200):
    step=size-overlap; return [t[i:i+size] for i in range(0,len(t),step)]

def sentence_chunks(t, n=6):
    s=nltk.sent_tokenize(t)
    return [" ".join(s[i:i+n]) for i in range(0,len(s),n)]

def paragraph_chunks(t, min_len=200):
    parts=[p.strip() for p in t.split("\n\n") if p.strip()]
    merged=[]; buf=""
    for p in parts:
        if len(p)<min_len: buf+=" "+p
        else:
            if buf: merged.append((buf+" "+p).strip()); buf=""
            else: merged.append(p)
    if buf: merged.append(buf)
    return merged

def semantic_chunks(t, win=6, thr=0.65):
    s=nltk.sent_tokenize(t)
    if not s: return []
    m=SentenceTransformer("all-MiniLM-L6-v2")
    e=m.encode(s, convert_to_numpy=True, show_progress_bar=False)
    e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-10)
    chunks=[]; i=0
    while i<len(s):
        j=min(i+win,len(s))
        while j<len(s):
            sim=float(np.dot(e[i:j].mean(axis=0), e[j]))
            if sim>=thr and (j-i)<win*3: j+=1
            else: break
        chunks.append(" ".join(s[i:j])); i=j
    return chunks

# -----------------------
# Retrieval (Top-3 Hybrid)
# -----------------------
def retrieve_top(chunks, q, k=5):
    try:
        m=SentenceTransformer("all-MiniLM-L6-v2")
        ce=m.encode(chunks,convert_to_numpy=True)
        qe=m.encode([q],convert_to_numpy=True)[0]
        sims=(ce@qe)/(np.linalg.norm(ce,axis=1)*np.linalg.norm(qe)+1e-10)
        idx=np.argsort(sims)[-k:][::-1]; return [chunks[i] for i in idx]
    except:
        scores=[sum(w.lower() in c.lower() for w in q.split()) for c in chunks]
        idx=np.argsort(scores)[-k:][::-1]; return [chunks[i] for i in idx]

# -----------------------
# Answer Prompt
# -----------------------
def answer_from_chunks(chunks, q):
    ctx="\n\n".join(chunks)
    prm=f"""
Use ONLY the context to answer.
If unclear, say: "Not clearly answered in the document."

Context:
{ctx}

Question: {q}

Answer in 4-7 bullet points with facts from the document:
"""
    return gemini_answer(prm)

# -----------------------
# Agentic Chunking
# -----------------------
def agentic_answer(text, q):
    # Normalize British ↔ US spelling for anemia
    text = text.replace("anaemia", "anemia")
    q = q.replace("anaemia", "anemia")

    # Chunk selection (same logic you had)
    c = paragraph_chunks(text, 200)
    if len(c) < 6:
        c = sentence_chunks(text, 6)
    if len(c) < 1:
        c = fixed_chunks(text, 1000)

    # ---- ✅ Embedding-based similarity instead of keyword match ----
    try:
        m = SentenceTransformer("all-MiniLM-L6-v2")
        ce = m.encode(c, convert_to_numpy=True)
        qe = m.encode([q], convert_to_numpy=True)[0]
        sims = (ce @ qe) / (np.linalg.norm(ce, axis=1) * np.linalg.norm(qe) + 1e-10)

        # take top 8 candidate chunks
        idx = np.argsort(sims)[-8:][::-1]
        cand = [c[i] for i in idx]
    except:
        # fallback if model fails
        scores = [sum(w.lower() in ch.lower() for w in q.split()) for ch in c]
        idx = np.argsort(scores)[-8:][::-1]
        cand = [c[i] for i in idx]

    # Prepare prompt for LLM to select best chunks
    numbered = "\n".join([f"{i+1}. {cand[i][:500]}" for i in range(len(cand))])
    sel = f"""
You are selecting the most relevant chunks to answer the question.

Return ONLY a JSON list of chunk numbers. No text explanation.

Question: {q}

Chunks:
{numbered}

Example output: [2,4,1]
"""

    r = gemini_answer(sel, 200)

    chosen = []
    try:
        arr = json.loads(re.search(r"\[.*?\]", r).group())
        for x in arr[:5]:
            if isinstance(x, int) and 1 <= x <= len(cand):
                chosen.append(cand[x-1])
    except:
        chosen = [cand[0]]

    return answer_from_chunks(chosen, q)

# -----------------------
# Sidebar
# -----------------------
method=st.sidebar.selectbox("Method",[
    "Fixed chunking",
    "Recursive chunking",
    "Semantic chunking",
    "Sentence chunking",
    "Paragraph chunking",
    "Agentic chunking"
])

st.sidebar.info(f"📄 PDF: `{PDF_PATH}`")

# -----------------------
# Main UI
# -----------------------
q=st.text_input("Ask a question about the PDF:")

if st.button("Get Answer"):
    if not q.strip(): st.warning("Enter a question."); st.stop()
    with st.spinner("Thinking..."):

        if method=="Fixed chunking":
            ch=fixed_chunks(doc_text)
        elif method=="Recursive chunking":
            ch=recursive_chunks(doc_text)
        elif method=="Semantic chunking":
            ch=semantic_chunks(doc_text)
        elif method=="Sentence chunking":
            ch=sentence_chunks(doc_text)
        elif method=="Paragraph chunking":
            ch=paragraph_chunks(doc_text)
        else: 
            ans=agentic_answer(doc_text,q)
            st.success("Answer (Agentic):")
            st.write(ans); st.stop()

        top=retrieve_top(ch,q,k=3)
        ans=answer_from_chunks(top,q)
        st.success(f"Answer ({method}):")
        st.write(ans)

st.caption("🔑 Gemini key + PDF path are set inside code")

           
