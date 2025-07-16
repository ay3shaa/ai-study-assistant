import fitz  # PyMuPDF
import re

import spacy
nlp = spacy.load("en_core_web_sm")

def chunk_by_sentence(text, max_tokens=500):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)



def clean_text(text):
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove extra spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


