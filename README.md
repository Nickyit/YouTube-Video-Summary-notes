# YouTube Video Summary tool
A web-based YouTube Video Summary tool that uses NLP and machine learning techniques to analyze video transcripts and generate short, informative summaries for efficient content consumption.

## Workflow
1. First, the audio is extracted from the YouTube video using `yt-dlp`.
2. Then, the audio is analyzed using an Automatic Speech Recognition (ASR) model (Whisper) to generate the transcript.
3. After that, the transcript text is processed by a Natural Language Processing (NLP) summarization model.
4. Finally, the summarized output is printed as the result.

In short:  
Video → Audio Extraction → Speech to Text (Transcript) → Text Summarization → Final Output
## YouTube Video Summary Tool

## Overview
This tool downloads the audio from a YouTube video, converts speech to text using OpenAI’s Whisper model, and produces an abstractive summary using a Transformer-based BART model. It is designed to help users quickly understand long video content.

## Tech Stack
- Python  
- yt-dlp (Audio Extraction)  
- OpenAI Whisper (Speech-to-Text)  
- HuggingFace Transformers – BART (Text Summarization)  
- Streamlit (Web UI)

## How to Run
```bash
pip install streamlit yt-dlp openai-whisper transformers torch
streamlit run app.py
