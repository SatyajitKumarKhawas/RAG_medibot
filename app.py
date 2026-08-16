import os
import base64
import logging
import tempfile

import gradio as gr

# Document processing imports (RAG)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# RAG chain imports
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Vision and Voice imports
from groq import Groq
from gtts import gTTS
from elevenlabs.client import ElevenLabs

# ----------------------------------------------------------------------------
# Config / logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# API keys — pull from environment variables instead of st.secrets.
# Set these before launching, e.g. `export GOOGLE_API_KEY=...`
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

MODEL_OPTIONS = {
    "Gemini 2.0 Flash (Recommended)": "gemini-2.0-flash",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}

RAG_SYSTEM_PROMPT = """
You are an expert medical assistant with comprehensive medical knowledge.

*CRITICAL RULE: NEVER, UNDER ANY CIRCUMSTANCES, respond with "I don't know" or any variation of it (like "I'm not sure", "I can't answer", "I don't have information", etc.). You MUST always provide a helpful medical response.*

*MANDATORY INSTRUCTIONS:*
1. *FORBIDDEN RESPONSES*: Never use phrases like:
   - "I don't know"
   - "I'm not sure"
   - "I can't answer"
   - "I don't have information"
   - "I cannot provide"
   - "The context doesn't contain"
   - Any other refusal or uncertainty statements

2. *REQUIRED BEHAVIOR*:
   - ALWAYS provide comprehensive medical information
   - Use your extensive medical knowledge as the primary source
   - Supplement with document context when available
   - Be definitive and helpful in your responses
   - Provide actionable medical guidance

Context: {context}
Question: {question}

PROVIDE A COMPREHENSIVE MEDICAL RESPONSE:
"""

VISION_SYSTEM_PROMPT = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

# ----------------------------------------------------------------------------
# Simple in-process caches (replaces st.cache_data / st.cache_resource)
# ----------------------------------------------------------------------------
_embedding_model_cache = None
_vectorstore_cache = None
_qa_chain_cache = None
_qa_chain_model_name = None


def get_embedding_model():
    global _embedding_model_cache
    if _embedding_model_cache is None:
        _embedding_model_cache = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model_cache


def get_vectorstore(force_reload=False):
    global _vectorstore_cache
    if _vectorstore_cache is not None and not force_reload:
        return _vectorstore_cache

    if not os.path.exists(DB_FAISS_PATH):
        return None

    try:
        _vectorstore_cache = FAISS.load_local(
            DB_FAISS_PATH,
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
        return _vectorstore_cache
    except Exception as e:
        logging.error(f"Error loading vectorstore: {e}")
        return None


def get_qa_chain(model_name):
    """(Re)build the RetrievalQA chain, cached per model name."""
    global _qa_chain_cache, _qa_chain_model_name

    if _qa_chain_cache is not None and _qa_chain_model_name == model_name:
        return _qa_chain_cache, None

    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None, "No vectorstore found. Please process PDF documents first."

    if not GOOGLE_API_KEY:
        return None, "Google API Key not found. Set the GOOGLE_API_KEY environment variable."

    try:
        prompt = PromptTemplate(template=RAG_SYSTEM_PROMPT, input_variables=["context", "question"])

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=GOOGLE_API_KEY,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        _qa_chain_cache = qa_chain
        _qa_chain_model_name = model_name
        return qa_chain, None
    except Exception as e:
        return None, f"Error setting up Gemini API: {e}"


# ----------------------------------------------------------------------------
# Document processing (PDF -> FAISS)
# ----------------------------------------------------------------------------
def load_pdf_files(data_path):
    if not os.path.exists(data_path):
        return [], f"Data directory '{data_path}' not found!"

    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents, None


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)


def create_vectorstore(text_chunks):
    embedding_model = get_embedding_model()
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db


def process_pdf_documents():
    """Gradio callback: process PDFs in DATA_PATH into the FAISS vectorstore."""
    documents, err = load_pdf_files(DATA_PATH)
    if err:
        return err
    if not documents:
        return "No PDF files found in the data directory!"

    text_chunks = create_chunks(documents)
    create_vectorstore(text_chunks)

    # Force the RAG chain to rebuild against the new vectorstore.
    global _qa_chain_cache, _qa_chain_model_name
    _qa_chain_cache = None
    _qa_chain_model_name = None
    get_vectorstore(force_reload=True)

    return f"Vectorstore created successfully with {len(text_chunks)} chunks!"


def vectorstore_status():
    return "✅ Vectorstore loaded successfully!" if os.path.exists(DB_FAISS_PATH) else "⚠️ No vectorstore found. Please process documents first."


# ----------------------------------------------------------------------------
# Vision helpers (GROQ)
# ----------------------------------------------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_query(query, model, encoded_image, api_key):
    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            ],
        }
    ]
    chat_completion = client.chat.completions.create(messages=messages, model=model)
    return chat_completion.choices[0].message.content


def analyze_image_with_text(image_path, user_query=""):
    if not GROQ_API_KEY:
        return "GROQ API key not set. Set the GROQ_API_KEY environment variable."
    try:
        encoded_image = encode_image(image_path)
        full_query = VISION_SYSTEM_PROMPT
        if user_query:
            full_query += f"\n\nUser's specific question: {user_query}"

        return analyze_image_with_query(
            query=full_query,
            encoded_image=encoded_image,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=GROQ_API_KEY,
        )
    except Exception as e:
        return f"Error analyzing image: {e}"


# ----------------------------------------------------------------------------
# Voice helpers (STT via GROQ, TTS via gTTS / ElevenLabs)
# ----------------------------------------------------------------------------
def transcribe_with_groq(audio_filepath, stt_model="whisper-large-v3"):
    if not GROQ_API_KEY:
        return "", "GROQ API key not set. Set the GROQ_API_KEY environment variable."
    if not audio_filepath:
        return "", "No audio recorded."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en",
            )
        return transcription.text, None
    except Exception as e:
        return "", f"Error transcribing audio: {e}"


def text_to_speech_with_gtts(input_text, output_filepath):
    gTTS(text=input_text, lang="en", slow=False).save(output_filepath)
    return output_filepath


def text_to_speech_with_elevenlabs(input_text, output_filepath, api_key):
    try:
        client = ElevenLabs(api_key=api_key)
        response = client.text_to_speech.convert(
            voice_id="Aria",
            model_id="eleven_turbo_v2",
            text=input_text,
        )
        with open(output_filepath, "wb") as f:
            for chunk in response:
                f.write(chunk)
        return output_filepath
    except Exception as e:
        logging.error(f"ElevenLabs TTS failed: {e}")
        return text_to_speech_with_gtts(input_text, output_filepath)


def generate_audio_response(text, use_elevenlabs=False):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            output_path = temp_audio.name

        if use_elevenlabs and ELEVENLABS_API_KEY:
            return text_to_speech_with_elevenlabs(text, output_path, ELEVENLABS_API_KEY)
        return text_to_speech_with_gtts(text, output_path)
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None


# ----------------------------------------------------------------------------
# Gradio callbacks
# ----------------------------------------------------------------------------
def handle_voice_input(audio_filepath):
    """Transcribe recorded/uploaded audio and hand the text back to the caller."""
    text, err = transcribe_with_groq(audio_filepath)
    if err:
        return gr.update(value=""), err
    return gr.update(value=text), "✅ Transcription complete — review and send below."


def handle_image_upload(image_path, model_display_name, use_elevenlabs):
    """Run vision analysis as soon as an image is uploaded."""
    if image_path is None:
        return None, None

    response_text = analyze_image_with_text(image_path)
    audio_path = generate_audio_response(response_text, use_elevenlabs=use_elevenlabs)
    return response_text, audio_path


def handle_chat(message, history, model_display_name):
    """RAG chat turn. `history` is the Gradio chatbot history (list of [user, bot])."""
    if not message or not message.strip():
        return history, "", ""

    model_name = MODEL_OPTIONS.get(model_display_name, "gemini-2.0-flash")
    qa_chain, err = get_qa_chain(model_name)

    if err:
        history = history + [[message, f"⚠️ {err}"]]
        return history, "", ""

    try:
        response = qa_chain.invoke({"query": message})
        result = response["result"]
        source_docs = response.get("source_documents", [])
    except Exception as e:
        result = f"Error generating response: {e}"
        source_docs = []

    sources_md = ""
    if source_docs:
        lines = [f"**Sources:** {len(source_docs)} document(s) referenced\n"]
        for i, doc in enumerate(source_docs, 1):
            snippet = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            lines.append(f"**Source {i}:**\n\n{snippet}\n")
        sources_md = "\n".join(lines)

    history = history + [[message, result]]
    return history, "", sources_md


def process_pdfs_and_refresh_status():
    msg = process_pdf_documents()
    return msg, vectorstore_status()


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------
with gr.Blocks(title="Unified Medical AI Assistant") as demo:
    gr.Markdown("# 🩺 Unified Medical AI Assistant — RAG + Vision + Voice")
    gr.Markdown("### 🤖 Smart routing: Text/Voice → RAG | Images → Vision Analysis")

    with gr.Accordion("⚙️ Configuration", open=False):
        gr.Markdown(
            "API keys are read from environment variables "
            "(`GOOGLE_API_KEY`, `GROQ_API_KEY`, `ELEVENLABS_API_KEY`) — "
            "set these before launching the app."
        )
        with gr.Row():
            google_key_display = gr.Textbox(
                label="Google API Key (Gemini)",
                value="●●●●●●●●" if GOOGLE_API_KEY else "Not set",
                interactive=False,
                type="password",
            )
            groq_key_display = gr.Textbox(
                label="GROQ API Key",
                value="●●●●●●●●" if GROQ_API_KEY else "Not set",
                interactive=False,
                type="password",
            )
            elevenlabs_key_display = gr.Textbox(
                label="ElevenLabs API Key (optional)",
                value="●●●●●●●●" if ELEVENLABS_API_KEY else "Not set",
                interactive=False,
                type="password",
            )

        use_elevenlabs = gr.Checkbox(
            label="Use ElevenLabs TTS (unchecked = free gTTS)",
            value=bool(ELEVENLABS_API_KEY),
        )

        model_select = gr.Dropdown(
            label="Choose Gemini Model",
            choices=list(MODEL_OPTIONS.keys()),
            value=list(MODEL_OPTIONS.keys())[0],
        )

        with gr.Row():
            process_btn = gr.Button("🔄 Process PDF Documents")
            vs_status = gr.Textbox(label="Vectorstore status", value=vectorstore_status(), interactive=False)

        process_result = gr.Textbox(label="Processing result", interactive=False)
        process_btn.click(fn=process_pdfs_and_refresh_status, outputs=[process_result, vs_status])

        gr.Markdown(
            "**How it works:**\n"
            "- Upload an image → Vision Analysis (GROQ)\n"
            "- Text/voice input → RAG Chatbot (Gemini)\n\n"
            "**Setup:** add API keys as environment variables, process PDF documents for RAG, "
            "then use voice, text, or images to interact."
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🎤 Voice Input")
            voice_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload audio")
            transcribe_btn = gr.Button("📝 Transcribe")
            voice_status = gr.Textbox(label="Voice status", interactive=False)
            voice_transcript = gr.Textbox(label="Transcript (edit if needed, then send below)")

        with gr.Column():
            gr.Markdown("## 📸 Image Upload")
            image_input = gr.Image(type="filepath", label="Upload medical image for analysis")
            vision_output = gr.Textbox(label="🖼️ Image Analysis Result", interactive=False)
            vision_audio_output = gr.Audio(label="Voice response", interactive=False)

    gr.Markdown("---")
    gr.Markdown("## 💬 Chat Interface")

    chatbot = gr.Chatbot(label="Medical RAG Chatbot", height=400)
    with gr.Row():
        chat_input = gr.Textbox(
            label="Ask medical questions",
            placeholder="Type a question, or paste a transcript from Voice Input above...",
            scale=4,
        )
        send_btn = gr.Button("Send", scale=1)

    sources_display = gr.Markdown(label="Source Documents")

    # Wiring
    transcribe_btn.click(
        fn=handle_voice_input,
        inputs=[voice_audio],
        outputs=[voice_transcript, voice_status],
    )

    image_input.change(
        fn=handle_image_upload,
        inputs=[image_input, model_select, use_elevenlabs],
        outputs=[vision_output, vision_audio_output],
    )

    send_btn.click(
        fn=handle_chat,
        inputs=[chat_input, chatbot, model_select],
        outputs=[chatbot, chat_input, sources_display],
    )
    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, chatbot, model_select],
        outputs=[chatbot, chat_input, sources_display],
    )

    gr.Markdown("---")
    gr.Markdown(
        "<div style='text-align: center; color: gray;'>"
        "⚠️ Disclaimer: This is for educational purposes only. "
        "Always consult a real healthcare professional for medical advice."
        "</div>"
    )

if __name__ == "__main__":
    demo.launch()
