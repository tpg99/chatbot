import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configurazione pagina ---
st.set_page_config(page_title="Chatbot Manuale BP")
st.title("Chatbot Manuale BP")
st.write("Fai una domanda basata sul file CSV con domande e risposte.")

# --- Caricamento modello ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Caricamento CSV ---
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    return df

# Percorso del CSV (deve essere nello stesso repo di app.py)
csv_path = "domande_risposte.csv"
df = load_csv(csv_path)

# --- Creazione embeddings ---
model = load_model()
questions = df["Domanda"].astype(str).tolist()
answers = df["Risposta"].astype(str).tolist()

embeddings = model.encode(questions, convert_to_tensor=False)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --- Interfaccia chat ---
user_input = st.text_input("Scrivi la tua domanda:")

if user_input:
    query_emb = model.encode([user_input])
    D, I = index.search(np.array(query_emb), k=3)
    
    st.markdown("### Risposte più rilevanti:")
    for i in I[0]:
        st.write(f"**Domanda simile:** {questions[i]}")
        st.write(f"**Risposta:** {answers[i]}")
        st.write("---")
