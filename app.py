import gradio as gr
import nltk
import string
import PyPDF2
import docx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# -------------------- NLTK SETUP --------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------- TEXT READERS --------------------
def read_text_file(file):
    if file is None:
        return ""

    filename = file.name.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8")

    elif filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif filename.endswith(".docx"):
        document = docx.Document(file)
        return "\n".join([p.text for p in document.paragraphs])

    else:
        return ""

# -------------------- NLP PREPROCESS --------------------
def preprocess_sentence(sentence):
    words = word_tokenize(sentence.lower())
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]
    return words

# -------------------- FREQUENCY SUMMARIZER --------------------
def frequency_based_summarizer(text, num_sentences):
    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return "❌ No valid text found in document."

    word_freq = {}
    for sentence in sentences:
        for word in preprocess_sentence(sentence):
            word_freq[word] = word_freq.get(word, 0) + 1

    sentence_scores = {}
    for sentence in sentences:
        for word in preprocess_sentence(sentence):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    summary_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:num_sentences]

    return " ".join(summary_sentences)

# -------------------- GRADIO FUNCTION --------------------
def summarize_uploaded_document(file, num_sentences):
    text = read_text_file(file)

    if len(text.strip()) == 0:
        return "❌ Uploaded file is empty or unsupported."

    return frequency_based_summarizer(text, num_sentences)

# -------------------- GRADIO UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Extractive Text Summarization App")
    gr.Markdown("Upload a document and get an extractive summary using NLP")

    file_input = gr.File(
        label="Upload Document",
        file_types=[".txt", ".pdf", ".docx"]
    )

    sentence_slider = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        label="Number of sentences in summary"
    )

    summarize_btn = gr.Button("Generate Summary")

    output_box = gr.Textbox(
        label="Summary",
        lines=15
    )

    summarize_btn.click(
        fn=summarize_uploaded_document,
        inputs=[file_input, sentence_slider],
        outputs=output_box
    )

# -------------------- LAUNCH --------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
