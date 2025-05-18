import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.document import Document
import warnings
import time
import re
import random

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv(override=True)

# Define constants
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
INDEX_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Page configuration
st.set_page_config(
    page_title="MediQuery - NHS Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app - updated system ready box to gray with dark text
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1lcbmhc, .css-1vs3yvj {
        background-color: #f1f7fe !important;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 1200px;
    }
    .info-box {
        background-color: #555555;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #333333;
        color: #ffffff;
    }
    .info-box h3 {
        color: #ffffff;
    }
    .results-area {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-box {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.9em;
        border-left: 3px solid #aaaaaa;
    }
    .citation {
        color: #0066cc;
        font-style: italic;
        font-size: 0.9em;
    }
    .bullet-point {
        margin-bottom: 15px;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
        margin-top: 2rem;
    }
    .stButton button {
        background-color: #2e86de;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    .loading-spinner {
        text-align: center;
        padding: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #7f8c8d;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'article_count' not in st.session_state:
    st.session_state.article_count = 0

# Sidebar
with st.sidebar:
    # st.image("https://www.nhs.uk/nhsuk-cms-theme/themes/nhsuk/assets/NHS-404-page.png", width=200)
    st.title("🏥")
    st.title("MediQuery")
    st.markdown("### Your AI NHS Medical Assistant")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    MediQuery uses advanced AI to answer your medical questions based on NHS articles. 
    
    All information is sourced directly from NHS UK's official content, ensuring accuracy and reliability.
    """)
    
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
    - 🔍 Instant medical information lookup
    - 📚 Based on official NHS articles
    - 📝 Detailed answers with citations
    - 🇬🇧 British medical terminology
    """)
    
    st.markdown("---")
    st.caption("This is a demonstration and not an official NHS product. Always consult healthcare professionals for medical advice.")

# Main content area
st.title("🏥 MediQuery - Ask NHS")
st.markdown("Ask any medical question and get information from NHS articles with proper citations.")

# Initialize RAG system
@st.cache_resource
def initialize_rag_system():
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_KEY")
        
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY or OPENAI_KEY environment variable.")
        return None, None, 0
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4.1", 
        temperature=0,
        openai_api_key=openai_api_key
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder="./embeddings_cache"
    )
    
    # Check if index exists
    if os.path.exists(INDEX_PATH) and os.path.isdir(INDEX_PATH) and len(os.listdir(INDEX_PATH)) > 0:
        try:
            vectorstore = Chroma(persist_directory=INDEX_PATH, embedding_function=embeddings)
            # Get count of articles
            df = pd.read_csv("nhs_articles.csv")
            article_count = len(df)
            return vectorstore, llm, article_count
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return None, None, 0
    else:
        # Create directory if it doesn't exist
        os.makedirs(INDEX_PATH, exist_ok=True)
        
        try:
            # Load data
            df = pd.read_csv("nhs_articles.csv")
            article_count = len(df)
            
            # Create documents with metadata
            documents = []
            for _, row in df.iterrows():
                content = (
                    f"Article Title: {row['title']}\n"
                    f"Article Link: {row['link']}\n"
                    f"Article Category: {row['category']}\n"
                    f"Article: {row['article']}"
                )
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": row['title'],
                        "link": row['link'],
                        "category": row['category']
                    }
                )
                documents.append(doc)
            
            # Text splitter for 2000 character chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                persist_directory=INDEX_PATH
            )
            
            # Save the vector store
            vectorstore.persist()
            
            return vectorstore, llm, article_count
        except Exception as e:
            st.error(f"Error building vector store: {e}")
            return None, None, 0

# Initialize system with loading indicator
with st.spinner("Initializing medical knowledge base..."):
    vectorstore, llm, article_count = initialize_rag_system()
    if vectorstore and llm:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        st.session_state.vector_store_ready = True
        st.session_state.article_count = article_count

# Display system status
if st.session_state.vector_store_ready:
    st.markdown(f"""
    <div class="info-box">
        <h3>📚 System Ready</h3>
        <p>Knowledge base loaded with {st.session_state.article_count} NHS articles.</p>
        <p>You can ask any medical question to get information with proper citations.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("System initialization failed. Please check your setup and API keys.")

# RAG Query Function
def process_medical_query(query):
    if not st.session_state.vector_store_ready:
        return "System is not initialized properly. Please check your setup."
    
    try:
        # Define the prompt template
        template = """
        You are a British medical research assistant. Your task is to answer only health-related questions based only on the context provided.
        You will be given a query and a context that contains information from NHS UK articles.
        Your response should be based on the context and should not include any information outside of it.
        You should always provide citations for the articles you refer to in your answer.

        Output Style:
        1. Answer the question by explaining the relevant context in bullet points.
        2. I fthe query is not related to health, let the user know.
        3. If the context does not contain enough information to answer the question, let the user know.
        4. If the context contains information that is not relevant to the question, let the user know.
        5. Strictly mention only the article title and link of the article as they are, as inline citation you are referring to from the context.
        6. If context is not relevant to the question, strictly let the user know.
        7. Do not change the article title and link in any way.
        
        Output format:
            1. Explain the relevant context in a conversational manner. This is the explanation of the context. (article title, link)
            2. You use only the relevant information from the context to answer the question. (article title, link)

        
        Context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        # Format docs function
        def format_docs(docs):
            context = ""
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                title = metadata.get("title", "Unknown")
                link = metadata.get("link", "Unknown")
                
                context += f"\nRelevant Article {i}:\nTitle: {title}\nLink: {link}\n{doc.page_content}\n\n"
            print(f"Formatted context: {context}")
            return context
        
        # Create and invoke RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        # Show typing effect
        response_placeholder = st.empty()
        with response_placeholder.container():
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate typing effect
            message_placeholder.markdown("Searching medical knowledge base...")
            time.sleep(1.5)
            
            # Get actual response
            response = rag_chain.invoke(query)
            full_response = response.content
            
            # Process response to highlight citations
            processed_response = re.sub(r'\((.*?), (https?://.*?)\)', r'<span class="citation">(\1, <a href="\2" target="_blank">source</a>)</span>', full_response)
            
            # Replace bullet points with styled divs
            bullet_pattern = r'• (.*?)(?=• |$)'
            if '• ' in processed_response:
                processed_response = re.sub(bullet_pattern, r'<div class="bullet-point">• \1</div>', processed_response, flags=re.DOTALL)
            
            # Display final response
            message_placeholder.markdown(processed_response, unsafe_allow_html=True)
        
        return full_response
    except Exception as e:
        return f"Error: {e}\nPlease try again or check your setup."

# Example questions
example_questions = [
    "What is PCOS and how does it affect fertility?",
    "What are the symptoms of diabetes?",
    "How can I manage my anxiety?",
    "What vaccinations are recommended for children?",
    "What treatments are available for migraines?",
    "What are the early signs of a stroke?"
]

# User Interface for Questions
st.markdown("### Ask a Medical Question")
st.markdown("Type your question or select one of the examples below.")

# Example question buttons
col1, col2 = st.columns(2)
with col1:
    if st.button(example_questions[0]):
        st.session_state.user_question = example_questions[0]
    if st.button(example_questions[2]):
        st.session_state.user_question = example_questions[2]
    if st.button(example_questions[4]):
        st.session_state.user_question = example_questions[4]

with col2:
    if st.button(example_questions[1]):
        st.session_state.user_question = example_questions[1]
    if st.button(example_questions[3]):
        st.session_state.user_question = example_questions[3]
    if st.button(example_questions[5]):
        st.session_state.user_question = example_questions[5]

# User input field
user_question = st.text_input("Your question:", value=st.session_state.get('user_question', ''))

# Search button
if st.button("Search NHS Knowledge Base", type="primary") and user_question:
    st.session_state.processing = True
    with st.spinner("Searching medical knowledge base..."):
        answer = process_medical_query(user_question)
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
        st.session_state.processing = False
    
    # Clear the input after submission
    st.session_state.user_question = ""

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Previous Questions")
    
    for i, exchange in enumerate(st.session_state.chat_history):
        with st.expander(f"Q: {exchange['question']}", expanded=(i == len(st.session_state.chat_history) - 1)):
            # Process response to highlight citations
            processed_response = re.sub(r'\((.*?), (https?://.*?)\)', r'<span class="citation">(\1, <a href="\2" target="_blank">source</a>)</span>', exchange['answer'])
            
            # Replace bullet points with styled divs
            bullet_pattern = r'• (.*?)(?=• |$)'
            if '• ' in processed_response:
                processed_response = re.sub(bullet_pattern, r'<div class="bullet-point">• \1</div>', processed_response, flags=re.DOTALL)
            
            st.markdown(processed_response, unsafe_allow_html=True)

# Help section
with st.expander("ℹ️ How to use MediQuery"):
    st.markdown("""
    ### How to Use MediQuery

    1. **Ask a Question**: Type your medical question in the text field or select one of the example questions.
    2. **Get Answers**: Click "Search NHS Knowledge Base" to get an answer based on NHS articles.
    3. **View Citations**: Each piece of information is cited with the original NHS article for reference.
    4. **Previous Questions**: Access your previous questions and answers in the expandable sections below.

    ### Tips for Good Questions

    - Be specific about symptoms, conditions, or treatments you're asking about
    - Include relevant details like age group or risk factors if applicable
    - Ask one question at a time for the most accurate answers
    
    Remember that MediQuery is for informational purposes only and does not replace professional medical advice.
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>MediQuery RAG System - Powered by LangChain and OpenAI</p>
    <p>Built with NHS data for educational purposes only.</p>
    <p>Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)