# app.py
"""
Chatbot Manuale BP
- Interfaccia Gradio in italiano
- Caricamento PDF (o uso del PDF pre-caricato)
- Estrazione testo dal PDF, chunking, embedding con all-MiniLM-L6-v2
- Indicizzazione con FAISS e ricerca semantica
- Tutto gratuito (modelli open-source)
"""

import os
from typing import List, Tuple
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# -------- Configurazione --------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400        # numero di caratteri per chunk
CHUNK_STEP = 200        # overlap tra chunk
INDEX_DIM = 384         # dim embedding di all-MiniLM-L6-v2

# -------- Utility: estrai testo dal PDF --------
def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = ""
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)

# -------- Utility: chunking semplice --------
def chunk_text(text: str, size: int = CHUNK_SIZE, step: int = CHUNK_STEP) -> List[str]:
    text = text.replace("\r", " ").replace("\n\n", "\n")
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks

# -------- Indice e embeddings (globali per semplicità) --------
model = SentenceTransformer(EMBED_MODEL_NAME)
index = None
chunks_store: List[Tuple[str, int]] = []  # lista di (chunk_text, original_chunk_index)

def build_index_from_pdf_bytes(pdf_bytes: bytes, filename_hint: str = "uploaded.pdf"):
    # salva temporaneamente
    tmp_path = f"/tmp/{filename_hint}"
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)
    text = extract_text_from_pdf(tmp_path)
    if not text.strip():
        raise ValueError("Impossibile estrarre testo dal PDF caricato.")
    chunks = chunk_text(text)
    if len(chunks) == 0:
        raise ValueError("PDF troppo corto o non contiene testo leggibile.")
    # embeddings
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    # crea indice FAISS
    global index, chunks_store
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    chunks_store = [(c, i) for i, c in enumerate(chunks)]
    # rimuovi file temporaneo
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return len(chunks)

# -------- Funzione di risposta --------
def answer_query(query: str, top_k: int = 1):
    if index is None or len(chunks_store) == 0:
        return "Carica prima un PDF (o inseriscilo nella cartella del progetto) per permettere al chatbot di rispondere."
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(q_emb), top_k)
    answers = []
    for idx in I[0]:
        if idx < 0 or idx >= len(chunks_store):
            continue
        chunk_text, _ = chunks_store[idx]
        answers.append(chunk_text)
    if not answers:
        return "Nessuna risposta trovata."
    # Unisci le migliori risposte per fornire contesto
    return "\n\n---\n\n".join(answers)

# -------- Gradio UI --------
with gr.Blocks(title="Chatbot Manuale BP") as demo:
    gr.Markdown("# Chatbot Manuale BP")
    gr.Markdown(
        "Carica un PDF contenente il manuale o le FAQ, poi poni una domanda. "
        "Il sistema restituisce porzioni del documento più pertinenti."
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_uploader = gr.File(label="Carica PDF (opzionale)", file_types=[".pdf"])
            btn_index = gr.Button("Indicizza PDF")
            status = gr.Label(value="Nessun PDF indicizzato", label="Stato")

            # Domande d'esempio
            examples = gr.Dropdown(
                choices=[
                    "BP non si apre o è lento",
                    "Non riesco a salvare un'anagrafica",
                    "Ho emesso una fattura su un cliente errato",
                ],
                label="Esempi rapidi"
            )
            btn_example = gr.Button("Usa esempio")
        with gr.Column(scale=2):
            query = gr.Textbox(lines=2, label="Fai una domanda sul manuale BP")
            submit = gr.Button("Chiedi")
            output = gr.Textbox(lines=10, label="Risposta")

    # Funzioni di callback
    def on_index_click(pdf_file):
        try:
            if pdf_file is None:
                # tenta caricare PDF locale presente nella cartella
                local_path = "Domande e risposte.pdf"
                if not os.path.exists(local_path):
                    return gr.update(value="Nessun PDF fornito e file locale non trovato.")
                with open(local_path, "rb") as f:
                    data = f.read()
                count = build_index_from_pdf_bytes(data, filename_hint="local.pdf")
                return gr.update(value=f"Indicizzazione completata: {count} chunk creati.")
            else:
                data = pdf_file.read()
                count = build_index_from_pdf_bytes(data, filename_hint=pdf_file.name)
                return gr.update(value=f"Indicizzazione completata: {count} chunk creati.")
        except Exception as e:
            return gr.update(value=f"Errore indicizzazione: {e}")

    def on_use_example(ex):
        return ex

    def on_ask(q):
        if not q or q.strip() == "":
            return "Inserisci prima una domanda."
        try:
            return answer_query(q, top_k=2)
        except Exception as e:
            return f"Errore nella ricerca: {e}"

    btn_index.click(on_index_click, inputs=[pdf_uploader], outputs=[status])
    btn_example.click(on_use_example, inputs=[examples], outputs=[query])
    submit.click(on_ask, inputs=[query], outputs=[output])

# Se esegui localmente: demo.launch()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
