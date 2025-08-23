import os
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

# Document processing imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# RAG chain imports
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Environment setup
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
        ">🗑️ Clear</button>
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
                document.getElementById('status').innerHTML = '⚠️ No speech detected';
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
            // Create a custom event to send the transcript to Streamlit
            const event = new CustomEvent('voiceInput', {
                detail: { transcript: finalTranscript.trim() }
            });
            window.dispatchEvent(event);
            
            // Also try to set the chat input value directly
            const chatInputs = parent.document.querySelectorAll('[data-testid="stChatInputTextArea"]');
            if (chatInputs.length > 0) {
                const chatInput = chatInputs[chatInputs.length - 1];
                chatInput.value = finalTranscript.trim();
                
                // Trigger input events
                chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                chatInput.dispatchEvent(new Event('change', { bubbles: true }));
            }
            
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

    // Listen for custom voice input events
    window.addEventListener('voiceInput', function(e) {
        console.log('Voice input received:', e.detail.transcript);
    });
    </script>
    """
    return voice_html

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
        
        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
You are an expert medical assistant with comprehensive medical knowledge. 

**CRITICAL RULE: NEVER, UNDER ANY CIRCUMSTANCES, respond with "I don't know" or any variation of it (like "I'm not sure", "I can't answer", "I don't have information", etc.). You MUST always provide a helpful medical response.**

**MANDATORY INSTRUCTIONS:**
1. **FORBIDDEN RESPONSES**: Never use phrases like:
   - "I don't know"
   - "I'm not sure"  
   - "I can't answer"
   - "I don't have information"
   - "I cannot provide"
   - "The context doesn't contain"
   - Any other refusal or uncertainty statements

2. **REQUIRED BEHAVIOR**: 
   - ALWAYS provide comprehensive medical information
   - Use your extensive medical knowledge as the primary source
   - Supplement with document context when available
   - Be definitive and helpful in your responses
   - Provide actionable medical guidance

3. **RESPONSE APPROACH**:
   - Start directly with medical information
   - Use both provided context AND your medical expertise
   - Give detailed, accurate medical explanations
   - Include relevant medical facts, symptoms, treatments, etc.
   - Always be informative and helpful

**Response Structure:**
---
**Medical Information:**
<Provide comprehensive medical answer using your knowledge + any relevant context>

**Key Medical Facts:**
• <Important fact 1>
• <Important fact 2>  
• <Important fact 3>

**Clinical Context:**
<Additional relevant medical information from your expertise>

**Document Reference (if applicable):**
- <Any relevant information from provided context>
  - Source: <document source if available>
---

**REMEMBER: You are a medical expert. Draw from your extensive medical training. Never claim ignorance. Always provide valuable medical information.**

Context:
{context}

Question:
{question}

PROVIDE A COMPREHENSIVE MEDICAL RESPONSE (NO "I DON'T KNOW" ALLOWED):
"""
        
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )
        
        # Setup Gemini LLM
        try:
            # Get model from session state or default
            model_name = getattr(st.session_state, 'selected_model', 'gemini-2.0-flash')
            
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                google_api_key=os.environ.get("GOOGLE_API_KEY")
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
            st.info("Please ensure GOOGLE_API_KEY is set in your .env file")
    
    def get_response(self, query):
        """Get response from QA chain"""
        if self.qa_chain is None:
            return "Sorry, the chatbot is not properly initialized. Please check your setup."
        
        try:
            response = self.qa_chain.invoke({'query': query})
            return response["result"], response["source_documents"]
        except Exception as e:
            return f"Error generating response: {str(e)}", []

def check_voice_input():
    """Check for voice input from session storage"""
    voice_input_js = """
    <script>
    const voiceInput = sessionStorage.getItem('voice_input');
    if (voiceInput) {
        // Clear the session storage
        sessionStorage.removeItem('voice_input');
        // Return the voice input
        return voiceInput;
    }
    return null;
    </script>
    """
    return components.html(voice_input_js, height=0)

def main():
    st.set_page_config(
        page_title="RAG Chatbot with Voice Input",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🏥 Medical RAG Chatbot with Voice Input & Gemini API")
    st.markdown("---")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Check if vectorstore exists
        vectorstore_exists = os.path.exists(DB_FAISS_PATH)
        
        if vectorstore_exists:
            st.success("✅ Vectorstore loaded successfully!")
        else:
            st.warning("⚠️ No vectorstore found. Please process documents first.")
        
        st.subheader("Process New Documents")
        
        if st.button("🔄 Process PDF Documents"):
            with st.spinner("Processing documents..."):
                # Load documents
                documents = DocumentProcessor.load_pdf_files(DATA_PATH)
                
                if not documents:
                    st.error("No PDF files found in the data directory!")
                    return
                
                st.info(f"Loaded {len(documents)} PDF pages")
                
                # Create chunks
                text_chunks = DocumentProcessor.create_chunks(documents)
                st.info(f"Created {len(text_chunks)} text chunks")
                
                # Create vectorstore
                DocumentProcessor.create_vectorstore(text_chunks)
                
                # Refresh the page to load new vectorstore
                st.rerun()
        
        st.subheader("🔧 Model Configuration")
        
        # Model selection
        model_options = {
            "Gemini 2.0 Flash (Recommended)": "gemini-2.0-flash", 
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Pro": "gemini-2.5-pro"
        }
        
        selected_model = st.selectbox(
            "Choose Gemini Model:",
            options=list(model_options.keys()),
            help="Different models offer various capabilities and speeds"
        )
        
        # Store selected model in session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_options[selected_model]
        
        if st.session_state.selected_model != model_options[selected_model]:
            st.session_state.selected_model = model_options[selected_model]
            # Reset chatbot to use new model
            if 'chatbot' in st.session_state:
                del st.session_state.chatbot
        
        st.markdown("---")
        st.subheader("🎤 Voice Input Info")
        st.markdown("""
        **Voice Input Features:**
        - 🎤 Click "Start Recording" to begin
        - 🛑 Click "Stop Recording" to end
        - 📤 Click "Send to Chat" to use transcribed text
        - 🗑️ Click "Clear" to reset
        
        **Supported Browsers:**
        - Chrome, Edge, Safari (latest versions)
        - Requires microphone permissions
        """)
        
        st.markdown("---")
        st.subheader("ℹ️ Setup Instructions")
        st.markdown("""
        **Available Models:**
        - **Gemini 2.0 Flash**: Latest with improved capabilities (Recommended)
        - **Gemini 2.5 Flash**: Best price-performance ratio
        - **Gemini 2.5 Pro**: Most powerful for complex medical reasoning
        
        **Setup Instructions:**
        1. Create a `.env` file with your Google API key:
           ```
           GOOGLE_API_KEY=your_api_key_here
           ```
        2. Place PDF files in the `data/` directory
        3. Click "Process PDF Documents" to create vectorstore
        4. Select your preferred model above
        5. Use voice input or text to ask medical questions!
        """)
    
    # Main chat interface
    if not os.path.exists(DB_FAISS_PATH):
        st.warning("Please process documents first using the sidebar.")
        return
    
    # Voice Input Component
    st.subheader("🎤 Voice Input")
    components.html(create_voice_input_component(), height=300)
    
    st.markdown("---")
    st.subheader("💬 Chat Interface")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask me any medical question about your documents... (or use voice input above)"):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message('assistant'):
            with st.spinner("Analyzing medical information..."):
                result, source_docs = st.session_state.chatbot.get_response(prompt)
                
                # Display result
                st.markdown(result)
                
                # Display source documents if available
                if source_docs:
                    with st.expander("📄 Source Documents"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.json(doc.metadata)
                            st.markdown("---")
                
                # Prepare content for session state
                content_with_sources = result
                if source_docs:
                    content_with_sources += f"\n\n**Sources:** {len(source_docs)} document(s) referenced"
                
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': content_with_sources
                })
    
    # Instructions for users
    st.markdown("---")
    st.info("""
    **How to use Voice Input:**
    1. Click the 🎤 "Start Recording" button above
    2. Speak your medical question clearly
    3. Click 🛑 "Stop Recording" when finished
    4. Review the transcribed text
    5. Click 📤 "Send to Chat" to submit your question
    
    **Tips for better voice recognition:**
    - Speak clearly and at a moderate pace
    - Use a quiet environment
    - Allow microphone permissions when prompted
    - You can edit the transcribed text before sending
    """)

# Standalone functions for direct usage (backward compatibility)
def load_llm_gemini():
    """Load Gemini LLM for standalone usage"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

def create_qa_chain_standalone():
    """Create QA chain for standalone usage"""
    # Load vectorstore
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    
    # Custom prompt
    CUSTOM_PROMPT_TEMPLATE = """
You are an expert medical assistant with comprehensive medical knowledge. 

**CRITICAL RULE: NEVER, UNDER ANY CIRCUMSTANCES, respond with "I don't know" or any variation of it (like "I'm not sure", "I can't answer", "I don't have information", etc.). You MUST always provide a helpful medical response.**

**MANDATORY INSTRUCTIONS:**
1. **FORBIDDEN RESPONSES**: Never use phrases like:
   - "I don't know"
   - "I'm not sure"  
   - "I can't answer"
   - "I don't have information"
   - "I cannot provide"
   - "The context doesn't contain"
   - Any other refusal or uncertainty statements

2. **REQUIRED BEHAVIOR**: 
   - ALWAYS provide comprehensive medical information
   - Use your extensive medical knowledge as the primary source
   - Supplement with document context when available
   - Be definitive and helpful in your responses
   - Provide actionable medical guidance

3. **RESPONSE APPROACH**:
   - Start directly with medical information
   - Use both provided context AND your medical expertise
   - Give detailed, accurate medical explanations
   - Include relevant medical facts, symptoms, treatments, etc.
   - Always be informative and helpful

**Response Structure:**
---
**Medical Information:**
<Provide comprehensive medical answer using your knowledge + any relevant context>

**Key Medical Facts:**
• <Important fact 1>
• <Important fact 2>  
• <Important fact 3>

**Clinical Context:**
<Additional relevant medical information from your expertise>

**Document Reference (if applicable):**
- <Any relevant information from provided context>
  - Source: <document source if available>
---

**REMEMBER: You are a medical expert. Draw from your extensive medical training. Never claim ignorance. Always provide valuable medical information.**

Context: {context}
Question: {question}

PROVIDE A COMPREHENSIVE MEDICAL RESPONSE (NO "I DON'T KNOW" ALLOWED):
"""
    
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm_gemini(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    
    return qa_chain

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        # This will only work if running via streamlit
        main()
    except:
        # Fallback to command line interface
        print("Medical RAG Chatbot - Command Line Mode")
        print("Make sure to set up your .env file with GOOGLE_API_KEY")
        
        # Process documents if needed
        if not os.path.exists(DB_FAISS_PATH):
            print("No vectorstore found. Processing documents...")
            documents = DocumentProcessor.load_pdf_files(DATA_PATH)
            if documents:
                text_chunks = DocumentProcessor.create_chunks(documents)
                DocumentProcessor.create_vectorstore(text_chunks)
                print("Vectorstore created successfully!")
            else:
                print("No documents found. Please add PDF files to the data/ directory.")
                exit()
        
        # Create QA chain and run
        qa_chain = create_qa_chain_standalone()
        
        while True:
            user_query = input("\nWrite Query Here (or 'quit' to exit): ")
            if user_query.lower() == 'quit':
                break
                
            response = qa_chain.invoke({'query': user_query})
            print("RESULT:", response["result"])
            print("SOURCE DOCUMENTS:", len(response["source_documents"]), "documents found")