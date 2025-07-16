import streamlit as st
import tempfile
from utils.parser import extract_text_from_pdf, chunk_by_sentence
from utils.retriever import build_faiss_index, retrieve_chunks
from utils.llm import ask_model, generate_flashcards

# Page config
st.set_page_config(page_title="AI Study Assistant", layout="wide")

# Sidebar Settings
with st.sidebar:
    st.title("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=1000, value=300, step=50)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
    st.markdown("---")
    

# Header
st.title("📘 AI Study Assistant")
st.markdown("Upload a PDF, ask questions, and generate flashcards using a **local LLM**.")

# PDF Upload
uploaded_file = st.file_uploader("📄 Upload your study PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("📄 Processing your PDF..."):
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_by_sentence(text)
        index = build_faiss_index(chunks)
        st.success(f"✅ PDF processed!")

    # Tabs for Question & Flashcards
    tab1, tab2 = st.tabs(["💬 Ask Questions", "🧠 Flashcards"])

    # --- Tab 1: Ask Questions ---
    with tab1:
        st.markdown("### 💬 Ask a question about your document")
        question = st.text_input("Type your question below:")

        if question:
            with st.spinner("🔍 Searching and generating answer..."):
                context = retrieve_chunks(question, chunks, index)
                answer = ask_model(question, context, max_tokens=max_tokens, temperature=temperature)
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("#### ✅ Answer:")
                    st.write(answer)

                with col2:
                    st.markdown("#### 📚 Context Used:")
                    st.code("\n\n---\n\n".join(context[:2]), language="markdown")

    # --- Tab 2: Flashcards ---
    with tab2:
        st.markdown("### 🧠 Generate Flashcards from your document")
        topic = st.text_input("Optional: Enter a topic ")

        if st.button("Generate Flashcards"):
            search_term = topic if topic else "overview"
            with st.spinner("📚 Generating flashcards..."):
                context = retrieve_chunks(search_term, chunks, index)
                cards = generate_flashcards(context,num_cards=3, max_tokens=max_tokens, temperature=temperature)
                st.markdown("#### 🃏 Flashcards:")
                st.code(cards, language="markdown")
