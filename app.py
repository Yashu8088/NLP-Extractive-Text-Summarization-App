# ===============================
# Extractive Text Summarization App
# ===============================

import gradio as gr
import nltk
import string
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import PyPDF2
import docx


# ===============================
# NLTK Downloads (Safe for HF)
# ===============================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# ===============================
# NLP Utilities
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_sentence(sentence):
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]
    return words


# ===============================
# File Readers
# ===============================
def read_text_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return ""


# ===============================
# Frequency-Based Summarizer
# ===============================
def frequency_based_summarizer(text, num_sentences=5):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    word_freq = Counter()

    for sentence in sentences:
        for word in preprocess_sentence(sentence):
            word_freq[word] += 1

    sentence_scores = {}

    for sentence in sentences:
        for word in preprocess_sentence(sentence):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    top_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:num_sentences]

    summary = " ".join(top_sentences)
    return summary


# ===============================
# Gradio App Logic
# ===============================
def summarize_uploaded_document(file, num_sentences):
    if file is None:
        return "❌ Please upload a document."

    text = read_text_file(file)

    if not text.strip():
        return "❌ The uploaded file is empty or unsupported."

    summary = frequency_based_summarizer(text, int(num_sentences))
    return summary


# ===============================
# Gradio UI (Blocks)
# ===============================
with gr.Blocks(title="Extractive Text Summarization App") as demo:

    gr.Markdown("# 📄 Extractive Text Summarization App")
    gr.Markdown(
        "Upload a document (TXT / PDF / DOCX) and generate an **extractive summary** "
        "using frequency-based NLP techniques."
    )

    file_input = gr.File(
        file_types=[".txt", ".pdf", ".docx"],
        label="Upload Document"
    )

    sentence_slider = gr.Slider(
        minimum=1,
        maximum=15,
        value=5,
        step=1,
        label="Number of Sentences in Summary"
    )

    summarize_button = gr.Button("Generate Summary")

    output_box = gr.Textbox(
        lines=15,
        label="Summary"
    )

    summarize_button.click(
        summarize_uploaded_document,
        inputs=[file_input, sentence_slider],
        outputs=output_box
    )

demo.launch()
