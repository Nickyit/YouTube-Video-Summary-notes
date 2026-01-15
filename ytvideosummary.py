import streamlit as st
import subprocess
import whisper
from transformers import pipeline, AutoTokenizer
import torch

st.set_page_config(page_title="YouTube Video Summarizer")
st.title("🎥 YouTube Video Summarizer")

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=model_name, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return summarizer, tokenizer

url = st.text_input("Enter YouTube Video URL")

if st.button("Generate Summary") and url:
    st.info("Downloading audio...")
    subprocess.run(["yt-dlp", "--no-playlist", "-x", "--audio-format", "mp3", "-o", "audio.mp3", url], check=True)

    st.info("Loading Whisper...")
    whisper_model = load_whisper()

    st.info("Transcribing...")
    result = whisper_model.transcribe("audio.mp3")
    transcript = result["text"]

    st.info("Loading Summarizer...")
    summarizer, tokenizer = load_summarizer()

    def chunk_by_tokens(text, max_tokens=900):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

    chunks = chunk_by_tokens(transcript)

    st.info("Summarizing...")
    final_summary = ""
    for chunk in chunks:
        summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
        final_summary += summary[0]["summary_text"] + " "

    st.success("Done!")
    st.text_area("Final Summary", final_summary, height=300)
