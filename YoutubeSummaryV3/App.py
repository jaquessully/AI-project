import streamlit as st
import subprocess
import os
from openai import OpenAI

# 🔐 GPT Setup (Only for summarization)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🎵 Step 1: Download audio using yt-dlp without conversion (avoid needing ffmpeg)
def download_audio(youtube_url, output_path="audio.webm"):
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "-o", output_path,
        youtube_url
    ]
    try:
        subprocess.run(command, check=True)
        st.success(f"✅ Audio downloaded successfully: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Download failed: {e}")
        return None

# 🎙️ Step 2: Transcribe audio via OpenAI Whisper API
def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        st.success("✅ Transcription complete.")
        return transcript.text
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return None

# 🤖 Step 3: Summarize via GPT (v1 syntax)
def summarize_text(transcript, custom_prompt):
    try:
        response = client.chat.completions.create(
            model= "gpt-4o",
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": transcript}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        summary = response.choices[0].message.content
        print("✅ Summarization complete.")
        return summary
    except Exception as e:
        print("❌ Summarization failed:", e)
        return None

def save_summary(summary_text, filename="youtube_summary.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary_text)
        return filename
    except Exception as e:
        st.error(f"Failed to save summary: {e}")
        return None

# Streamlit UI
st.title("🎥 YouTube Video Summarizer")

url = st.text_input("Paste YouTube Video URL")

if st.button("Summarize"):
    with st.spinner("🌐 Downloading audio..."):
        audio_file = download_audio(url)

    if audio_file and os.path.exists(audio_file):
        with st.spinner("🎤 Transcribing audio via API..."):
            transcript = transcribe_audio(audio_file)

        if transcript:
            custom_prompt = """
[CONTEXT]

You are an expert summarizer for panel discussions in business and tech with strong attention to context, structure, and clarity.

[LOGICAL GOAL]
Provide a structured summary of Youtube video transcripts that highlights:
1. Panelists and their backgrounds
2. Key topics in order of appearance, with timestamps
3. Main takeaways: extract essentials tips, framework, and insights
4. Books, companies, or key people referenced

Key idea is to spot concepts, trends, risks and opportunities I could add to my knowledge slip box.

[EXPLICIT CONSTRAINTS]
- Use bullet points
- Max 4 lines per point
- Include timestamps in (mm:ss) format
- Keep each section under 250 words
- For references, if data missing, say so - do not invent

[CRITICAL INSIGHT TAGGING]
- Flag key sentences, quotes, or statistics worth turning into atomic notes
- Prefix each flagged sentence with 👉
- Ideal tags include strategic insight, risk signals, frameworks, first-principle thinking, and strong metaphors

[ACTION BREAKDOWN]
Break the response into these sections:
- Panelists
- Topics Covered
- Takeaways
- Referenced People, Companies, Books, Tech concepts

[RESPONSE TEMPLATE]
Format:
```
## Panelists
- Name (LinkedIn): Short bio

## Topics Covered
- **[Topic Title] (Timestamp)**  
  - One-line summary of the idea
  - Reasoning or key arguments
  👉 Quote or stat for Zettelkasten

## Takeaways
- Bullet insights

## References
- Bullet format: Company, Person, Book, tech concepts (with context)
```

[TONE CALIBRATION]
Be analytical, structured, and pragmatic. Do not flatter. Do not speculate. Prioritize clarity and signal.
"""
            with st.spinner("🤖 Summarizing transcript using GPT-4o..."):
                summary = summarize_text(transcript, custom_prompt)

            if summary:
                filename = save_summary(summary)
                if filename:
                    with open(filename, "r", encoding="utf-8") as file:
                        st.text_area("📄 Summary Output", file.read(), height=600)
                    st.download_button(
                        label="🔍 Download Summary as TXT",
                        data=open(filename, "rb"),
                        file_name=filename,
                        mime="text/plain"
                    )

        os.remove(audio_file)
