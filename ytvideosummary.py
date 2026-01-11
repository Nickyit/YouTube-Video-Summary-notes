import subprocess
import whisper
from transformers import pipeline

url = "https://www.youtube.com/watch?v=UpH88I9XsiY"

# Download audio
subprocess.run([
    "yt-dlp",
    "--no-playlist",
    "-x", "--audio-format", "mp3",
    "-o", "audio.mp3",
    url
], check=True)

# Load Whisper model (Speech → Text)
model = whisper.load_model("base")
result = model.transcribe("audio.mp3")

transcript = result["text"]

# print("\n===== FULL TRANSCRIPT =====\n")
# print(transcript)

# Load Summarization model (Text → Summary)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)

print("\n===== SUMMARY =====\n")
print(summary[0]["summary_text"])
