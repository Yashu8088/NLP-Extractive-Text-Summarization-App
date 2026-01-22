import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import string
import os

import PyPDF2
import docx

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# ---------- FILE READING ----------
def read_uploaded_file(file):
    if file is None:
        return ""

    file_path = file.name
    ext = os.path.splitext(file_path)[1].lower()

    # TXT
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # PDF
    elif ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    # DOCX
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        return ""


# ---------- SUMMARIZER ----------
def frequency_based_summarizer(text, num_sentences=4):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    words = [
        w for w in words
        if w not in stop_words and w not in string.punctuation
    ]

    word_freq = Counter(words)

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

    summary_sentences = sorted(
        sentence_scores, key=sentence_scores.get, reverse=True
    )[:num_sentences]

    return " ".join(summary_sentences)


def summarize_file(file):
    text = read_uploaded_file(file)

    if not text.strip():
        return "❌ Please upload a valid TXT, PDF, or DOCX file."

    return frequency_based_summarizer(text)


# ---------- GRADIO UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Extractive Text Summarization App")
    gr.Markdown("Upload a document and get the most important sentences.")

    file_input = gr.File(
        label="Upload Document",
        file_types=[".txt", ".pdf", ".docx"]
    )

    output_text = gr.Textbox(
        lines=8,
        label="Summary"
    )

    summarize_btn = gr.Button("Summarize")

    summarize_btn.click(
        fn=summarize_file,
        inputs=file_input,
        outputs=output_text
    )


if __name__ == "__main__":
    demo.launch(share=True)
