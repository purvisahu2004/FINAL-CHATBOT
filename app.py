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
        try: nltk.download("punkt_tab")
        except: pass

ensure_nltk_tokenizer()

# -----------------------------
# CONFIG
# -----------------------------
GEMINI_API_KEY = "AIzaSyBnNcI3bPJWgdbbTGBqqRr3tM9hSWYvWaQ"
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
    if not ok: return msg

    model = genai.GenerativeModel(MODEL, generation_config={"max_output_tokens": max_tokens})
    res = model.generate_content(prompt)
    return res.text

# -----------------------------
# Load PDF
# -----------------------------
def read_pdf(path):
    if not os.path.exists(path): return ""
    t = ""
    for p in PdfReader(path).pages:
        content = p.extract_text()
        if content: t += content + "\n"
    return t.strip()

doc_text = read_pdf(PDF_PATH)
if not doc_text:
    st.error("❌ PDF not found.")
    st.stop()

# -----------------------------
# IMPROVED Chunk Functions
# -----------------------------
def fixed_chunks(t, size=1000):
    t=t.replace("\n"," "); return [t[i:i+size] for i in range(0,len(t),size)]

def recursive_chunks(t, size=1000, ov=200):
    step=size-ov; return [t[i:i+size] for i in range(0,len(t),step)]

def sentence_chunks(t, n=5):
    s=nltk.sent_tokenize(t); return [" ".join(s[i:i+n]) for i in range(0,len(s),n)]

def paragraph_chunks(t, target_chunk_size=600):
    """Improved paragraph chunking that creates reasonable number of chunks"""
    # First split by paragraphs
    paragraphs = [x.strip() for x in t.split("\n\n") if x.strip() and len(x.strip()) > 50]
    
    # If we have very few paragraphs, use sentence chunks instead
    if len(paragraphs) <= 3:
        return sentence_chunks(t, n=8)
    
    merged = []
    current_chunk = ""
    
    for para in paragraphs:
        # If current chunk is empty, start with this paragraph
        if not current_chunk:
            current_chunk = para
        # If adding this paragraph would exceed target size, save current chunk
        elif len(current_chunk) + len(para) > target_chunk_size:
            merged.append(current_chunk.strip())
            current_chunk = para
        else:
            # Add paragraph to current chunk
            current_chunk += " " + para
    
    # Don't forget the last chunk
    if current_chunk:
        merged.append(current_chunk.strip())
    
    # If we still have too few chunks, merge some together
    if len(merged) < 5:
        # Re-merge with larger target size
        return paragraph_chunks(t, target_chunk_size=800)
    
    return merged

def semantic_chunks(t, win=6, thr=0.65):
    s=nltk.sent_tokenize(t)
    if len(s) == 0:
        return [t]
    m=SentenceTransformer("all-MiniLM-L6-v2")
    e=m.encode(s, convert_to_numpy=True); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-10)
    out=[]; i=0
    while i<len(s):
        j=min(i+win,len(s))
        while j<len(s):
            if float(np.dot(e[i:j].mean(axis=0), e[j]))>=thr: j+=1
            else: break
        out.append(" ".join(s[i:j])); i=j
    return out

# -----------------------------
# Retrieve Top-K chunks ✅ (now k=5) - OPTIMIZED
# -----------------------------
def retrieve_top(chunks, q, k=5):
    if len(chunks) <= k:
        return chunks[:k]  # ✅ FIX: Always return max k chunks
    
    try:
        m=SentenceTransformer("all-MiniLM-L6-v2")
        ce=m.encode(chunks,convert_to_numpy=True)
        qe=m.encode([q],convert_to_numpy=True)[0]
        sims=(ce@qe)/(np.linalg.norm(ce,axis=1)*np.linalg.norm(qe)+1e-10)
        idx=np.argsort(sims)[-k:][::-1]
        return [chunks[i] for i in idx]
    except:
        scores=[sum(w.lower() in c.lower() for w in q.split()) for c in chunks]
        idx=np.argsort(scores)[-k:][::-1]
        return [chunks[i] for i in idx]

# -----------------------------
# Prompt
# -----------------------------
def answer_from_chunks(chunks, q):
    ctx="\n\n".join(chunks)
    prm=f"""
Use ONLY this context to answer.
If unclear, say: "Not clearly answered in the document."

Context:
{ctx}

Question: {q}

Answer in 5-7 bullet points with facts from the document:
"""
    return gemini_answer(prm)

# -----------------------------
# OPTIMIZED Agentic Method - Much Faster
# -----------------------------
def agentic_answer(text, q):
    text=text.replace("anaemia","anemia"); q=q.replace("anaemia","anemia")
    
    # Create chunks only once
    chunks = paragraph_chunks(text)
    if len(chunks) < 6: 
        chunks = sentence_chunks(text, 8)
    if len(chunks) < 1: 
        chunks = fixed_chunks(text, 1000)
    
    # Get top candidates using similarity
    top_candidates = retrieve_top(chunks, q, k=8)
    
    # Simple heuristic selection instead of LLM call for speed
    query_words = set(q.lower().split())
    scored_chunks = []
    
    for i, chunk in enumerate(top_candidates):
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        # Bonus for longer matches
        scored_chunks.append((score + len(chunk)/10000, i, chunk))
    
    # Sort by score and take top 5
    scored_chunks.sort(reverse=True)
    chosen = [chunk for _, _, chunk in scored_chunks[:5]]
    
    return answer_from_chunks(chosen, q), chosen

# -----------------------------
# UI - FIXED with BETTER DEBUG INFO
# -----------------------------
method = st.sidebar.selectbox("Method",[
    "Fixed chunking","Recursive chunking","Semantic chunking",
    "Sentence chunking","Paragraph chunking","Agentic chunking"
])

q = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not q.strip(): 
        st.warning("Enter a question")
        st.stop()

    with st.spinner("Thinking..."):
        # ---- Get Chunks ----
        if method == "Fixed chunking": 
            chunks = fixed_chunks(doc_text)
            top_chunks = retrieve_top(chunks, q, k=5)
            ans = answer_from_chunks(top_chunks, q)
            
        elif method == "Recursive chunking": 
            chunks = recursive_chunks(doc_text)
            top_chunks = retrieve_top(chunks, q, k=5)
            ans = answer_from_chunks(top_chunks, q)
            
        elif method == "Semantic chunking": 
            chunks = semantic_chunks(doc_text)
            top_chunks = retrieve_top(chunks, q, k=5)
            ans = answer_from_chunks(top_chunks, q)
            
        elif method == "Sentence chunking": 
            chunks = sentence_chunks(doc_text)
            top_chunks = retrieve_top(chunks, q, k=5)
            ans = answer_from_chunks(top_chunks, q)
            
        elif method == "Paragraph chunking": 
            chunks = paragraph_chunks(doc_text)
            
            # Debug information
            with st.expander("📊 Debug Info - Paragraph Chunking"):
                st.write(f"**Total chunks created:** {len(chunks)}")
                if len(chunks) > 0:
                    st.write(f"**Chunk sizes:** {[len(chunk) for chunk in chunks]}")
                    st.write(f"**Average chunk size:** {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
                    st.write("**First few chunks preview:**")
                    for i, chunk in enumerate(chunks[:3]):
                        st.write(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:200]}...")
            
            # ✅ FORCE only top 5 chunks
            top_chunks = retrieve_top(chunks, q, k=5)
            
            # If we don't have enough chunks, use fallback
            if len(chunks) < 3:
                st.warning("⚠️ Few chunks created, using sentence chunks as fallback")
                chunks = sentence_chunks(doc_text, n=6)
                top_chunks = retrieve_top(chunks, q, k=5)
            
            ans = answer_from_chunks(top_chunks, q)
            
        else:  # Agentic chunking
            ans, top_chunks = agentic_answer(doc_text, q)
            st.success("Answer (Agentic)")
            st.write(ans)

            # ✅ Show chunks used
            with st.expander("🔍 Show Evidence (Top Selected Chunks)"):
                for i, ck in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(ck)
                    st.markdown("---")
            st.stop()

        # ---- For non-agentic methods ----
        st.success(f"Answer ({method})")
        st.write(ans)

        # ✅ Show top-5 chunks on demand
        with st.expander("🔍 Show Top 5 Evidence Chunks"):
            for i, ck in enumerate(top_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(ck[:500] + "..." if len(ck) > 500 else ck)
                st.markdown("---")

st.caption("✅ Click 'Show Evidence' to view the chunks used.")
