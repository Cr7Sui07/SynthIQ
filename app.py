# app.py
import streamlit as st
import tempfile
import fitz  # PyMuPDF
import whisper
import openai
import os
import time
from langdetect import detect
from googletrans import Translator

# Load your OpenAI API key securely from .streamlit/secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]
translator = Translator()

st.set_page_config(layout="wide", page_title="AI Training Assistant", page_icon="✨")

# Progress Meter
def loading_bar(text, duration=2):
    with st.spinner(text):
        for percent in range(0, 101, 5):
            time.sleep(duration / 20)
            st.progress(percent)

# File upload
st.markdown("## Upload Training Material (PDF or Video)")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "mp4", "mkv"])

content_text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_type) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if file_type == "pdf":
        st.info("Extracting text from PDF...")
        loading_bar("Reading PDF")
        doc = fitz.open(tmp_path)
        content_text = " ".join(page.get_text() for page in doc)
    else:
        st.info("Transcribing video using Whisper...")
        loading_bar("Processing Video")
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        content_text = result["text"]

    # Detect language and translate if needed
    lang = detect(content_text[:500])
    if lang != "en":
        st.info("Translating to English...")
        loading_bar("Translating")
        content_text = translator.translate(content_text, src=lang, dest='en').text

    st.success("Content processed successfully!")

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Quiz", "Scenarios", "AI Tutor"])

    with tab1:
        st.header("Summary")
        with st.spinner("Generating summary..."):
            try:
                short_text = content_text[:3000]
                summary = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Summarize the following text:\n\n{short_text}"}]
                )
                st.write(summary['choices'][0]['message']['content'])
            except Exception as e:
                st.error("An error occurred while generating the summary.")
                st.exception(e)

    with tab2:
        st.header("Quiz Time")
        with st.spinner("Creating quiz questions..."):
            try:
                quiz_prompt = f"Create 5 MCQs with 4 options each based on this content. Mark the correct answer with (*) symbol:\n{content_text[:3000]}"
                quiz = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": quiz_prompt}]
                )["choices"][0]["message"]["content"]
                st.markdown(quiz)
            except Exception as e:
                st.error("Failed to create quiz.")
                st.exception(e)

    with tab3:
        st.header("Situational Practice")
        with st.spinner("Creating scenario-based questions..."):
            try:
                scenario_prompt = f"Generate 3 real-world training scenarios from this content. Ask the user how they would respond."
                scenarios = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": scenario_prompt}]
                )["choices"][0]["message"]["content"]
                st.markdown(scenarios)
            except Exception as e:
                st.error("Failed to generate scenarios.")
                st.exception(e)

    with tab4:
        st.header("AI Chat Tutor")
        st.info("Ask questions about your material")
        user_q = st.text_input("Ask me anything from your uploaded content")
        if user_q:
            try:
                chat_prompt = f"This is the content: {content_text}\nNow answer the user's question: {user_q}"
                reply = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": chat_prompt}]
                )["choices"][0]["message"]["content"]
                st.success(reply)
            except Exception as e:
                st.error("Failed to answer the question.")
                st.exception(e)
