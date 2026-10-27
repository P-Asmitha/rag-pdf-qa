import os, io, time
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from google import genai
from google.genai import types

st.set_page_config(page_title="Ask Your PDF", page_icon="📄")
st.title("📄 Ask Your PDF")
st.caption("Upload a PDF, ask a question, and get an answer from the document with the sources shown.")

CHUNK_SIZE = 500   # characters in each piece
STEP = 400         # each piece starts 400 characters after the last, so pieces overlap by 100
TOP_K = 5          # how many pieces are sent to the AI
MODELS = ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3-flash-preview"]

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_index(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        return [], None
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), STEP)]
    vectors = load_model().encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors).astype("float32"))
    return chunks, index

api_key = st.sidebar.text_input("Gemini API key", type="password",
                                value=os.environ.get("GEMINI_API_KEY", ""))

uploaded = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about the PDF")

if st.button("Get answer"):
    if not api_key.strip():
        st.error("Paste your Gemini API key in the left sidebar.")
    elif uploaded is None:
        st.error("Upload a PDF first.")
    elif not question.strip():
        st.error("Type a question.")
    else:
        with st.spinner("Reading the PDF..."):
            chunks, index = build_index(uploaded.getvalue())
        if not chunks:
            st.error("No text found in this PDF. It may be a scanned image.")
        else:
            q = load_model().encode([question])
            k = min(TOP_K, len(chunks))
            _, ids = index.search(np.array(q).astype("float32"), k)
            top_chunks = [chunks[i] for i in ids[0]]
            context = "\n---\n".join(top_chunks)

            prompt = f"""Answer the question using ONLY the notes below.
If the answer is not in the notes, say "Not found in the document."

Notes:
{context}

Question: {question}"""

            client = genai.Client(
                api_key=api_key.strip(),
                http_options=types.HttpOptions(timeout=30000),
            )
            response = None
            last_error = ""
            with st.spinner("Asking the AI..."):
                for attempt in range(1, 3):
                    for name in MODELS:
                        try:
                            response = client.models.generate_content(model=name, contents=prompt)
                            break
                        except Exception as e:
                            last_error = str(e)[:200]
                    if response is not None:
                        break
                    time.sleep(5)

            if response is None:
                st.error(f"The AI is busy or the key is not working. Try again in a minute. Details: {last_error}")
            else:
                st.subheader("Answer")
                st.write(response.text)
                with st.expander("Sources used"):
                    for c in top_chunks:
                        st.write(c)
                        st.divider()
