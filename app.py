import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Chatbot Manuale BP",
    page_icon="🤖",
    layout="centered"
)

# --- LOGO AZIENDALE ---
# Assicurati che il file 'logo.png' sia nella stessa cartella del progetto
if os.path.exists("logo.png"):
    st.image("logo.png", width=200)
else:
    st.markdown("### 🤖 Chatbot Manuale BP")

st.write("Fai una domanda basata sul file CSV con domande, risposte e referenti.")

# --- SFONDO PERSONALIZZATO ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #e6f0ff 0%, #ffffff 100%);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- CARICAMENTO MODELLO ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- CARICAMENTO CSV ---
@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

csv_file = "domande_risposte_final.csv"
df = None

if os.path.exists(csv_file):
    st.success(f"✅ File CSV trovato: {csv_file}")
    df = load_csv_from_path(csv_file)
else:
    st.warning("⚠️ File CSV non trovato. Caricalo manualmente:")
    uploaded_file = st.file_uploader(
        "Carica un file CSV (con colonne 'domanda', 'risposta', 'chi_interpellare')",
        type="csv"
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV caricato con successo!")
    else:
        st.stop()

# --- VERIFICA COLONNE ---
expected_cols = {"domanda", "risposta", "chi_interpellare"}
if not expected_cols.issubset(df.columns):
    st.error(f"❌ Il CSV deve contenere le colonne {expected_cols}. Colonne trovate: {list(df.columns)}")
    st.stop()

if len(df) == 0:
    st.error("❌ Il CSV è vuoto! Aggiungi almeno una riga con domanda e risposta.")
    st.stop()

# --- CREAZIONE EMBEDDINGS ---
questions = df["domanda"].astype(str).tolist()
answers = df["risposta"].astype(str).tolist()
contacts = df["chi_interpellare"].astype(str).tolist()

st.info(f"📚 Domande caricate: {len(questions)}")

embeddings = model.encode(questions, convert_to_tensor=False)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --- GESTIONE STATO ---
if "reset" not in st.session_state:
    st.session_state.reset = False

# --- INTERFACCIA CHAT ---
st.subheader("💬 Inserisci la tua domanda:")
if not st.session_state.reset:
    user_input = st.text_input("Domanda:")
else:
    st.session_state.reset = False
    user_input = st.text_input("Domanda:", value="", key="new_question")

if user_input:
    query_emb = model.encode([user_input])
    D, I = index.search(np.array(query_emb), k=3)

    st.markdown("### 📖 Risposte trovate:")
    for i in I[0]:
        st.markdown(f"**Domanda simile:** {questions[i]}")
        st.markdown(f"**Risposta:** {answers[i]}")
        if contacts[i] and contacts[i].strip():
            st.markdown(f"**Chi interpellare:** {contacts[i]}")
        st.markdown("---")

    # Bottone per nuova domanda
    if st.button("🔄 Fai un'altra domanda"):
        st.session_state.reset = True
        st.rerun()
