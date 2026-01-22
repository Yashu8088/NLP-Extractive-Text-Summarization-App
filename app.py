import gradio as gr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


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

def read_text_file(file):
    if file is None:
        return ""
    
    with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
# Main Summarization Function (APP Logic)
def summarize_uploaded_document(file):
    text = read_text_file(file)

    if len(text.strip()) == 0:
        return "please upload a valid text file."

    summary = frequency_based_summarizer(text, num_sentences=5)

    return summary      
interface = gr.Interface(
    fn=summarize_uploaded_document,
    inputs=gr.File(label="Upload a Text Document (.txt)"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="Extractive Text Summarization App",
    description="Upload a document and automatically generate a concise summary using NLP-based extractive summarization."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

