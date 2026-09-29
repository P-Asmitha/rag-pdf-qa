# 📄 RAG-Based PDF Question Answering

A Retrieval-Augmented Generation (RAG) app that answers questions from an uploaded PDF, grounded strictly in the document's content.

## How it works
1. **Read** — extracts text from the uploaded PDF using `pypdf`.
2. **Chunk** — splits the text into overlapping 500-character pieces so no sentence is cut without a copy.
3. **Embed** — converts each chunk into a vector using `Sentence Transformers` (`all-MiniLM-L6-v2`).
4. **Retrieve** — stores vectors in a `FAISS` index and finds the 5 chunks closest to the question.
5. **Generate** — sends those chunks and the question to the Gemini API, instructed to answer only from them.
6. **Show sources** — displays the exact passages used, so the answer is verifiable, not hallucinated.

## Tech stack
Python, Streamlit, pypdf, Sentence Transformers, FAISS, Google Gemini API

## Run it locally
