import os
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components
import base64
import logging
import tempfile
import platform
import subprocess

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
import elevenlabs
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# System prompts
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

# Voice input HTML/JS component
def create_voice_input_component():
    """Create the voice input HTML component"""
    voice_html = """
    <div style="padding: 10px; border: 2px dashed #ccc; border-radius: 10px; margin: 10px 0; text-align: center;">
        <h4 style="margin-top: 0;">🎤 Voice Input</h4>
        <button id="startBtn" onclick="startRecording()" style="
            background-color: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">🎤 Start Recording</button>
        
        <button id="stopBtn" onclick="stopRecording()" disabled style="
            background-color: #f44336; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">🛑 Stop Recording</button>
        
        <div id="status" style="margin: 10px; font-weight: bold; color: #666;"></div>
        <div id="transcript" style="
            margin: 10px; 
            padding: 10px; 
            background-color: #f0f0f0; 
            border-radius: 5px; 
            min-height: 40px;
            font-style: italic;
        ">Your transcribed text will appear here...</div>
        
        <button id="sendBtn" onclick="sendToChat()" disabled style="
            background-color: #2196F3; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">📤 Send to Chat</button>
        
        <button id="clearBtn" onclick="clearTranscript()" style="
            background-color: #ff9800; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        ">🗑 Clear</button>
    </div>

    <script>
    let recognition = null;
    let isRecording = false;
    let finalTranscript = '';

    // Check if browser supports speech recognition
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
        recognition = new SpeechRecognition();
    }

    if (recognition) {
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            isRecording = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('status').innerHTML = '🔴 Recording... Speak now!';
            document.getElementById('status').style.color = '#f44336';
        };

        recognition.onresult = function(event) {
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }
            
            document.getElementById('transcript').innerHTML = 
                finalTranscript + '<span style="color: #999;">' + interimTranscript + '</span>';
        };

        recognition.onerror = function(event) {
            document.getElementById('status').innerHTML = '❌ Error: ' + event.error;
            document.getElementById('status').style.color = '#f44336';
            resetButtons();
        };

        recognition.onend = function() {
            isRecording = false;
            resetButtons();
            if (finalTranscript.trim() !== '') {
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('status').innerHTML = '✅ Recording completed!';
                document.getElementById('status').style.color = '#4CAF50';
            } else {
                document.getElementById('status').innerHTML = '⚠ No speech detected';
                document.getElementById('status').style.color = '#ff9800';
            }
        };
    } else {
        document.getElementById('status').innerHTML = '❌ Speech recognition not supported in this browser';
        document.getElementById('startBtn').disabled = true;
    }

    function startRecording() {
        if (recognition && !isRecording) {
            finalTranscript = '';
            document.getElementById('transcript').innerHTML = 'Listening...';
            document.getElementById('sendBtn').disabled = true;
            recognition.start();
        }
    }

    function stopRecording() {
        if (recognition && isRecording) {
            recognition.stop();
        }
    }

    function resetButtons() {
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }

    function sendToChat() {
        if (finalTranscript.trim() !== '') {
            // Store in session storage for Streamlit to pick up
            parent.sessionStorage.setItem('voice_input', finalTranscript.trim());
            
            document.getElementById('status').innerHTML = '📤 Sent to chat!';
            document.getElementById('status').style.color = '#4CAF50';
        }
    }

    function clearTranscript() {
        finalTranscript = '';
        document.getElementById('transcript').innerHTML = 'Your transcribed text will appear here...';
        document.getElementById('sendBtn').disabled = true;
        document.getElementById('status').innerHTML = '';
        parent.sessionStorage.removeItem('voice_input');
    }
    </script>
    """
    return voice_html

# Utility Functions
@st.cache_data
def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_query(query, model, encoded_image, api_key):
    """Analyze image using GROQ API."""
    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    
    return chat_completion.choices[0].message.content

def transcribe_with_groq(stt_model, audio_filepath, api_key):
    """Transcribe audio using GROQ API."""
    client = Groq(api_key=api_key)
    
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    
    return transcription.text

def text_to_speech_with_gtts(input_text, output_filepath):
    """Convert text to speech using gTTS."""
    language = "en"
    
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    return output_filepath




def text_to_speech_with_elevenlabs(input_text, output_filepath, api_key):
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=api_key)

        response = client.text_to_speech.convert(
            voice_id="Aria",  # change to valid voice
            model_id="eleven_turbo_v2",
            text=input_text
        )

        with open(output_filepath, "wb") as f:
            for chunk in response:
                f.write(chunk)

        return output_filepath
    except Exception as e:
        st.error(f"ElevenLabs TTS failed: {e}")
        return text_to_speech_with_gtts(input_text, output_filepath)





class DocumentProcessor:
    """Handles PDF loading and processing"""
    
    @staticmethod
    def load_pdf_files(data_path):
        """Load PDF files from directory"""
        if not os.path.exists(data_path):
            st.error(f"Data directory '{data_path}' not found!")
            return []
            
        loader = DirectoryLoader(
            data_path,
            glob='*.pdf',
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return documents
    
    @staticmethod
    def create_chunks(documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(documents)
        return text_chunks
    
    @staticmethod
    def create_vectorstore(text_chunks):
        """Create and save FAISS vectorstore"""
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Create vectorstore directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        
        st.success(f"Vectorstore created successfully with {len(text_chunks)} chunks!")
        return db



class RAGChatbot:
    """Main RAG chatbot class"""
    
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.setup_chain()

    
    @st.cache_resource
    def get_vectorstore(_self):
        """Load vectorstore with caching"""
        if not os.path.exists(DB_FAISS_PATH):
            return None
            
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        try:
            db = FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            return db
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            return None
    
    def setup_chain(self):
        """Setup the QA chain"""
        self.vectorstore = self.get_vectorstore()
        
        if self.vectorstore is None:
            return
        
        prompt = PromptTemplate(
            template=RAG_SYSTEM_PROMPT, 
            input_variables=["context", "question"]
        )
        
        # Setup Gemini LLM
        try:
            # Get model from session state or default
            model_name = getattr(st.session_state, 'selected_model', 'gemini-2.0-flash')
            # Use st.secrets to retrieve the API key securely
            google_api_key = st.secrets.get("GOOGLE_API_KEY")

            if not google_api_key:
                st.error("Google API Key not found in Streamlit Secrets.")
                return

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                google_api_key=google_api_key
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': prompt}
            )
            
        except Exception as e:
            st.error(f"Error setting up Gemini API: {str(e)}")
    
    def get_response(self, query):
        """Get response from QA chain"""
        if self.qa_chain is None:
            return "Sorry, the chatbot is not properly initialized. Please check your setup.", []
        
        try:
            response = self.qa_chain.invoke({'query': query})
            return response["result"], response["source_documents"]
        except Exception as e:
            return f"Error generating response: {str(e)}", []


class VisionProcessor:
    """Handles image analysis and vision processing"""

    def __init__(self, groq_api_key, elevenlabs_api_key=None):
        # Pass keys from the main function
        self.groq_api_key = groq_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
    
    def analyze_image_with_text(self, image_path, user_query=""):
        """Analyze image with optional user query"""
        try:
            encoded_image = encode_image(image_path)
            full_query = VISION_SYSTEM_PROMPT
            if user_query:
                full_query += f"\n\nUser's specific question: {user_query}"
            
            response = analyze_image_with_query(
                query=full_query,
                encoded_image=encoded_image,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                api_key=self.groq_api_key
            )
            return response
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def generate_audio_response(self, text, use_elevenlabs=False):
        """Generate audio response from text"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                output_path = temp_audio.name
            
            if use_elevenlabs and self.elevenlabs_api_key:
                text_to_speech_with_elevenlabs(text, output_path, self.elevenlabs_api_key)
            else:
                text_to_speech_with_gtts(text, output_path)
            
            return output_path
        except Exception as e:
            st.error(f"Error generating audio: {e}")
            return None

def check_voice_input():
    """Check for voice input from session storage"""
    voice_input_js = """
    <script>
    const voiceInput = sessionStorage.getItem('voice_input');
    if (voiceInput) {
        sessionStorage.removeItem('voice_input');
        return voiceInput;
    }
    return null;
    </script>
    """
    return components.html(voice_input_js, height=0)

def main():
    st.set_page_config(
        page_title="Unified Medical AI Assistant",
        page_icon="🩺",
        layout="wide"
    )
    
    st.title("🩺 Unified Medical AI Assistant - RAG + Vision + Voice")
    st.markdown("### 🤖 Smart routing: Text/Voice → RAG | Images → Vision Analysis")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙ Configuration")
        
        # API Keys
        st.subheader("🔑 API Keys")
        
        # Retrieve keys from Streamlit's secrets for display and use
        google_api_key = st.secrets.get("GOOGLE_API_KEY")
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        elevenlabs_api_key = st.secrets.get("ELEVENLABS_API_KEY")
        
        # These text inputs are for display only, to show users if the keys are set
        st.text_input(
            "Google API Key (Gemini):",
            value=google_api_key,
            type="password",
            help="Add this key to Streamlit Cloud Secrets"
        )
        
        st.text_input(
            "GROQ API Key:",
            value=groq_api_key,
            type="password",
            help="Add this key to Streamlit Cloud Secrets"
        )
        
        st.text_input(
            "ElevenLabs API Key (Optional):",
            value=elevenlabs_api_key,
            type="password",
            help="Add this key to Streamlit Cloud Secrets"
        )
        
        use_elevenlabs = st.checkbox(
            "Use ElevenLabs TTS", 
            value=bool(elevenlabs_api_key),
            help="Uncheck to use free gTTS instead"
        )
        
        st.markdown("---")
        
        # RAG Configuration
        st.subheader("📚 RAG Configuration")
        vectorstore_exists = os.path.exists(DB_FAISS_PATH)
        
        if vectorstore_exists:
            st.success("✅ Vectorstore loaded successfully!")
        else:
            st.warning("⚠ No vectorstore found. Please process documents first.")
        
        if st.button("🔄 Process PDF Documents"):
            with st.spinner("Processing documents..."):
                documents = DocumentProcessor.load_pdf_files(DATA_PATH)
                
                if not documents:
                    st.error("No PDF files found in the data directory!")
                else:
                    text_chunks = DocumentProcessor.create_chunks(documents)
                    DocumentProcessor.create_vectorstore(text_chunks)
                    st.rerun()
        
        # Model selection
        model_options = {
            "Gemini 2.0 Flash (Recommended)": "gemini-2.0-flash", 
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Pro": "gemini-2.5-pro"
        }
        
        selected_model = st.selectbox(
            "Choose Gemini Model:",
            options=list(model_options.keys())
        )
        
        # Store selected model in session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_options[selected_model]
        
        if st.session_state.selected_model != model_options[selected_model]:
            st.session_state.selected_model = model_options[selected_model]
            if 'rag_chatbot' in st.session_state:
                del st.session_state.rag_chatbot
        
        st.markdown("---")
        st.subheader("ℹ How It Works")
        st.markdown("""
        *🎯 Smart Routing:*
        - *Upload Image* → Vision Analysis (GROQ)
        - *Text/Voice Input* → RAG Chatbot (Gemini)
        
        *📋 Setup:*
        1. Add API keys above
        2. Process PDF documents for RAG
        3. Use voice, text, or images to interact
        """)
    
    # Check API keys
    if not google_api_key:
        st.error("Please provide your Google API key in the Streamlit Cloud secrets for RAG functionality.")
    
    if not groq_api_key:
        st.error("Please provide your GROQ API key in the Streamlit Cloud secrets for vision analysis.")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎤 Voice Input")
        components.html(create_voice_input_component(), height=280)
    
    with col2:
        st.header("📸 Image Upload")
        uploaded_image = st.file_uploader(
            "Upload medical image for analysis", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload triggers Vision Analysis (bypasses RAG)"
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    # Initialize components
    if google_api_key and 'rag_chatbot' not in st.session_state:
        vectorstore_exists = os.path.exists(DB_FAISS_PATH)
        if vectorstore_exists:
            st.session_state.rag_chatbot = RAGChatbot()
    
    if groq_api_key and 'vision_processor' not in st.session_state:
        st.session_state.vision_processor = VisionProcessor(groq_api_key, elevenlabs_api_key)
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Check for voice input
    voice_input = None
    try:
        if st.button("🔄 Check Voice Input", help="Click to check if voice input is available"):
            pass
    except:
        pass
    
    st.markdown("---")
    st.header("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message.get('type') == 'image_analysis':
                st.markdown("🖼 *Image Analysis Result:*")
            st.markdown(message['content'])
            
            # Display audio if available
            if message.get('audio_path') and os.path.exists(message['audio_path']):
                st.audio(message['audio_path'], format="audio/mp3")
    
    # Process uploaded image immediately
    if uploaded_image and groq_api_key:
        st.markdown("### 🔍 Processing Image...")
        
        with st.spinner("Analyzing image..."):
            try:
                # Save uploaded image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                    temp_img.write(uploaded_image.read())
                    temp_img_path = temp_img.name
                
                # Analyze image (Vision path - bypasses RAG)
                vision_response = st.session_state.vision_processor.analyze_image_with_text(temp_img_path)
                
                # Generate audio response
                audio_path = None
                if use_elevenlabs:
                    audio_path = st.session_state.vision_processor.generate_audio_response(
                        vision_response, use_elevenlabs=True
                    )
                else:
                    audio_path = st.session_state.vision_processor.generate_audio_response(
                        vision_response, use_elevenlabs=False
                    )
                
                # Add to chat
                st.session_state.messages.append({
                    'role': 'user',
                    'content': f"📸 Uploaded image: {uploaded_image.name}",
                    'type': 'image_upload'
                })
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': vision_response,
                    'type': 'image_analysis',
                    'audio_path': audio_path
                })
                
                # Clean up temp file
                os.unlink(temp_img_path)
                
                # Reset uploaded image to prevent reprocessing
                st.session_state.uploaded_image_processed = True
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Chat input for text/voice
    if prompt := st.chat_input("Ask medical questions (text input) or use voice input above..."):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Process with RAG (text input path)
        if 'rag_chatbot' in st.session_state and vectorstore_exists:
            with st.chat_message('assistant'):
                with st.spinner("Analyzing medical information..."):
                    result, source_docs = st.session_state.rag_chatbot.get_response(prompt)
                    
                    # Display result
                    st.markdown(result)
                    
                    # Display source documents if available
                    if source_docs:
                        with st.expander("📄 Source Documents"):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"*Source {i}:*")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.json(doc.metadata)
                                st.markdown("---")
                    
                    # Prepare content for session state
                    content_with_sources = result
                    if source_docs:
                        content_with_sources += f"\n\n*Sources:* {len(source_docs)} document(s) referenced"
                    
                    st.session_state.messages.append({
                        'role': 'assistant', 
                        'content': content_with_sources,
                        'type': 'rag_response'
                    })
        else:
            st.error("RAG chatbot not available. Please check your configuration and ensure documents are processed.")
    
    # Instructions
    st.markdown("---")
    st.info("""
    *🎯 Smart Usage Guide:*
    
    *For Vision Analysis (Image + AI Doctor):*
    - Upload any medical image above
    - System automatically uses GROQ vision model
    - Get instant AI doctor analysis with voice response
    
    *For RAG Chatbot (Text + Documents):*
    - Type questions in the chat or use voice input
    - System searches your uploaded PDF documents
    - Get comprehensive answers from your medical database
    
    *Voice Input:*
    - Click 🎤 "Start Recording" → Speak → "Stop Recording" → "Send to Chat"
    - Works with both RAG and Vision modes
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "⚠ Disclaimer: This is for educational purposes only. Always consult a real healthcare professional for medical advice."
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()