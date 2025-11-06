import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2

# --- Configurazione iniziale ---
st.set_page_config(page_title="Chatbot Manuale BP")
st.title("Chatbot Manuale BP")
st.write("Fai una domanda basata sul manuale PDF incluso nell’app.")

# --- Caricamento modello e PDF ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

model = load_model()
text = load_pdf("Domande_e_risposte.pdf")

# --- Creazione embedding ---
sentences = [s.strip() for s in text]()
