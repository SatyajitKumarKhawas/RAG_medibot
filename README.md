# 🩺 Unified Medical AI Assistant

### RAG + Vision + Voice Powered Medical Assistant

An advanced AI-powered medical assistant built with **Streamlit**, combining:

* 📚 **RAG (Retrieval-Augmented Generation)** using Gemini
* 📸 **Vision Analysis** using GROQ Vision Models
* 🎤 **Voice Input & Audio Responses**
* 🔊 **Text-to-Speech (gTTS / ElevenLabs)**

---

## 🚀 Features

### 1️⃣ RAG-Based Medical Chatbot

* Upload medical PDFs
* Automatically creates FAISS vector database
* Uses **Gemini (Google Generative AI)** for intelligent answers
* Returns source document references

### 2️⃣ Medical Image Analysis

* Upload medical images (X-ray, skin condition, etc.)
* Uses GROQ Vision Model (`llama-4-scout`)
* Provides concise doctor-style analysis
* Generates audio response

### 3️⃣ Voice Interaction

* Browser-based speech recognition
* Convert speech → text
* Convert AI response → audio (MP3)
* Optional ElevenLabs high-quality voice

### 4️⃣ Smart Routing

| Input Type | Model Used   |
| ---------- | ------------ |
| Text       | RAG (Gemini) |
| Voice      | RAG (Gemini) |
| Image      | GROQ Vision  |

---

## 🏗️ Tech Stack

* **Frontend**: Streamlit
* **LLM (RAG)**: Google Gemini (ChatGoogleGenerativeAI)
* **Vision Model**: GROQ (LLaMA 4 Scout)
* **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
* **Vector Store**: FAISS
* **Speech-to-Text**: GROQ Whisper
* **Text-to-Speech**: gTTS / ElevenLabs
* **Document Processing**: LangChain

---

## 📂 Project Structure

```
project/
│
├── data/                    # Upload your PDF medical documents here
├── vectorstore/
│   └── db_faiss/            # Auto-created FAISS vector database
│
├── app.py                   # Main Streamlit application
├── requirements.txt
└── README.md
```

---

## 🔑 API Keys Required

Add these in **Streamlit Cloud → Secrets** or `.streamlit/secrets.toml`

```toml
GOOGLE_API_KEY = "your_gemini_key"
GROQ_API_KEY = "your_groq_key"
ELEVENLABS_API_KEY = "your_elevenlabs_key"
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/unified-medical-ai.git
cd unified-medical-ai
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add Secrets

Create:

```
.streamlit/secrets.toml
```

Add your API keys.

### 4️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📚 How to Use

### 📝 For RAG Chatbot

1. Place medical PDFs inside `/data`
2. Click **"Process PDF Documents"**
3. Ask medical questions in chat
4. Get answers + source references

---

### 📸 For Vision Analysis

1. Upload a medical image
2. System automatically analyzes using GROQ
3. Get AI doctor response
4. Listen to audio explanation

---

### 🎤 For Voice Input

1. Click 🎤 Start Recording
2. Speak clearly
3. Stop Recording
4. Send to Chat

---

## 🧠 Model Options

Available Gemini Models:

* `gemini-2.0-flash` (Recommended)
* `gemini-2.5-flash`
* `gemini-2.5-pro`

---

## 🔊 Audio Modes

| Mode       | Quality | Cost |
| ---------- | ------- | ---- |
| gTTS       | Basic   | Free |
| ElevenLabs | Premium | Paid |

---

## ⚠ Disclaimer

This application is built **for educational and research purposes only**.
It is NOT a replacement for professional medical advice.
Always consult a licensed healthcare provider.

---

## 🌟 Future Improvements

* Multi-language support
* Medical report generation (PDF export)
* Patient history tracking
* Fine-tuned medical embeddings
* Deployment with Docker
* Role-based authentication

---

## 👨‍💻 Author

Developed as a unified AI healthcare assistant integrating:

* Retrieval-Augmented Generation
* Vision AI
* Voice AI


