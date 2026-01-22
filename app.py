import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_API_INFO"] = "False"


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

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

def frequency_based_summarizer(text, num_sentences=5):
    sentences = sent_tokenize(text)

    word_freq = defaultdict(int)
    for sentence in sentences:
        for word in preprocess_sentence(sentence):
            word_freq[word] += 1

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in preprocess_sentence(sentence):
            sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]

    top_indices = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:num_sentences]

    top_indices.sort()
    summary = " ".join([sentences[i] for i in top_indices])
    return summary

from PyPDF2 import PdfReader
from docx import Document


def read_document(file):
    if file is None:
        return ""

    file_path = file.name
    ext = os.path.splitext(file_path)[1].lower()

    text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text

# Main Summarization Function (APP Logic)

def summarize_uploaded_document(file, num_sentences):
    # Safety check
    if file is None:
        return "Please upload a document."

    # Read text from uploaded file
    text = read_document(file)

    # Validate extracted text
    if not text or len(text.strip()) == 0:
        return "Unable to extract text. Please upload a valid TXT, PDF, or DOCX file."

    # Generate summary
    summary = frequency_based_summarizer(
        text=text,
        num_sentences=num_sentences
    )

    return summary
    
interface = gr.Interface(
    fn=summarize_uploaded_document,
    inputs=[
        gr.File(
            file_types=[".txt", ".pdf", ".docx"],
            label="Upload Document"
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of sentences in summary"
        )
    ],
    outputs=gr.Textbox(
        lines=15,
        label="Summary"
    ),
    title="Extractive Text Summarization App",
    description="Upload a document and generate an extractive summary using frequency-based NLP"
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


