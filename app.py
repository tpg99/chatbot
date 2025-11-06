import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- Configurazione pagina ---
st.set_page_config(page_title="Chatbot Manuale BP")
st.title("Chatbot Manuale BP")
st.write("Fai una domanda basata sul file CSV con domande e risposte.")

# --- Caricamento modello ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- Funzione di caricamento CSV ---
@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

# --- Prova a caricare CSV locale oppure permetti upload ---
csv_file = "domande_risposte.csv"
df = None

if os.path.exists(csv_file):
    st.success(f"✅ File CSV trovato: {csv_file}")
    df = load_csv_from_path(csv_file)
else:
    st.warning("⚠️ File CSV non trovato nel progetto. Carica un file CSV manualmente:")
    uploaded_file = st.file_uploader("Carica un file CSV (deve avere colonne 'Domanda' e 'Risposta')", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV caricato con successo!")
    else:
        st.stop()

# --- Verifica formato ---
if not {"Domanda", "Risposta"}.issubset(df.columns):
    st.error("❌ Il CSV deve contenere le colonne 'Domanda' e 'Risposta'.")
    st.stop()

# --- Creazione embedding ---
questions = df["Domanda"].astype(str).tolist()
answers = df["Risposta"].astype(str).tolist()

st.info(f"📚 Domande caricate: {len(questions)}")

embeddings = model.encode(questions, convert_to_tensor=False)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --- Interfaccia chat ---
user_input = st.text_input("💬 Scrivi la tua domanda:")

if user_input:
    query_emb = model.encode([user_input])
    D, I = index.search(np.array(query_emb), k=3)
    
    st.markdown("### Risposte più rilevanti:")
    for i in I[0]:
        st.write(f"**Domanda simile:** {questions[i]}")
        st.write(f"**Risposta:** {answers[i]}")
        st.write("---")
