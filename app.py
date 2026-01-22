import gradio as gr
import nltk
import string
import PyPDF2
import docx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

# =========================
# Download NLTK resources
# =========================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))


# =========================
# File Reading Function
# =========================
def read_text_file(file):
    """
    Gradio File component returns a file path (NamedString),
    NOT a file object.
    """
    if file is None:
        return ""

    file_path = str(file).lower()

    # TXT
    if file_path.endswith(".txt"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # PDF
    if file_path.endswith(".pdf"):
        text = ""
        with open(file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    # DOCX
    if file_path.endswith(".docx"):
        document = docx.Document(file)
        return " ".join(p.text for p in document.paragraphs)

    return ""


# =========================
# Frequency-based Summarizer
# =========================
def frequency_based_summarizer(text, num_sentences=5):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    words = word_tokenize(text.lower())
    words = [
        w for w in words
        if w not in STOP_WORDS and w not in string.punctuation
    ]

    word_freq = Counter(words)

    sentence_scores = {}
    for sent in sentences:
        sent_words = word_tokenize(sent.lower())
        for word in sent_words:
            if word in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

    summary_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:num_sentences]

    return " ".join(summary_sentences)


# =========================
# Main Summarization Logic
# =========================
def summarize(file):
    text = read_text_file(file)

    if not text.strip():
        return "❌ Please upload a valid TXT, PDF, or DOCX file."

    summary = frequency_based_summarizer(text, num_sentences=5)
    return summary


# =========================
# Gradio UI
# =========================
demo = gr.Interface(
    fn=summarize,
    inputs=gr.File(
        label="Upload Document (TXT / PDF / DOCX)",
        file_types=[".txt", ".pdf", ".docx"]
    ),
    outputs=gr.Textbox(
        label="Summary",
        lines=10
    ),
    title="📄 Document Summarization App",
    description="Extractive text summarization using NLP (NLTK + Frequency-based method)"
)


# =========================
# Run App
# =========================
if __name__ == "__main__":
    demo.launch()
