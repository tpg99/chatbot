import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2

# --- Caricamento modello e dati ---
st.title("Chatbot Manuale BP")
st.write("Fai una domanda basata sul PDF caricato.")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Caricamento PDF
uploaded_file = st.file_uploader("Carica un file PDF", type="pdf")

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("✅ PDF caricato correttamente!")
    
    # Creiamo gli embeddings
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 5]
    embeddings = model.encode(sentences, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    # Chat
    question = st.text_input("Scrivi la tua domanda:")
    if question:
        q_emb = model.encode([question])
        D, I = index.search(np.array(q_emb), k=1)
        st.write("**Risposta:**", sentences[I[0][0]])
