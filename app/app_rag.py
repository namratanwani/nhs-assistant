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
from datetime import datetime

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
    page_title="Chat with NHS - Medical Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'article_count' not in st.session_state:
    st.session_state.article_count = 0
if 'num_docs' not in st.session_state:
    st.session_state.num_docs = 5
if 'show_rephrased' not in st.session_state:
    st.session_state.show_rephrased = False

# Sidebar
with st.sidebar:
    st.title("🏥 NHS Chat Assistant")
    
    st.markdown("---")
    st.markdown("### Settings")
    
    # Document retrieval control
    num_docs = st.slider(
        "Documents to retrieve:",
        min_value=2,
        max_value=15,
        value=st.session_state.num_docs,
        help="More documents = more context but slower"
    )
    st.session_state.num_docs = num_docs
    
    # Show rephrased query option
    show_rephrased = st.checkbox(
        "Show rephrased queries",
        value=st.session_state.show_rephrased,
        help="See how queries are optimized"
    )
    st.session_state.show_rephrased = show_rephrased
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Chat stats
    if st.session_state.messages:
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Get medical information from official NHS articles with AI-powered search and contextual understanding.")

# Main title
st.title("💬 Chat with NHS")

# Initialize RAG system
@st.cache_resource
def initialize_rag_system():
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY or OPENAI_KEY environment variable.")
        return None, None, 0
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
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
            
            # Text splitter
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
            
            vectorstore.persist()
            return vectorstore, llm, article_count
        except Exception as e:
            st.error(f"Error building vector store: {e}")
            return None, None, 0

# Query rephrasing function
def rephrase_query_with_context(original_query, messages):
    """Simple GPT-based query rephrasing"""
    if not messages:
        return original_query, False
    
    # Get recent context (last 2 exchanges)
    recent_messages = messages[-4:] if len(messages) >= 4 else messages
    context = ""
    for i in range(0, len(recent_messages), 2):
        if i+1 < len(recent_messages):
            context += f"Q: {recent_messages[i]['content']}\nA: {recent_messages[i+1]['content'][:200]}...\n\n"
    
    prompt = f"""Given this conversation context:
{context}

Rephrase this query to be standalone and clear: "{original_query}"

Only return the rephrased query, nothing else."""

    try:
        response = llm.invoke(prompt)
        rephrased = response.content.strip().strip('"')
        return rephrased, True
    except:
        return original_query, False

# Initialize system
with st.spinner("Initializing medical knowledge base..."):
    vectorstore, llm, article_count = initialize_rag_system()
    if vectorstore and llm:
        st.session_state.vector_store_ready = True
        st.session_state.article_count = article_count
        st.success(f"✅ Ready! Loaded {article_count} NHS articles")

# RAG Query Function
def process_medical_query(query):
    if not st.session_state.vector_store_ready:
        return "System not ready. Please check setup.", None, False
    
    try:
        # Rephrase query if needed
        rephrased_query, was_rephrased = rephrase_query_with_context(query, st.session_state.messages)
        
        # Update retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": st.session_state.num_docs}
        )
        
        # Get relevant documents using rephrased query
        relevant_docs = retriever.invoke(rephrased_query)
        
        # Create context
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            metadata = doc.metadata
            title = metadata.get("title", "Unknown")
            link = metadata.get("link", "Unknown")
            context += f"\nArticle {i}:\nTitle: {title}\nLink: {link}\n{doc.page_content}\n\n"
        
        # Define prompt
        template = """You are a British medical assistant. Answer the health question using only the NHS articles provided.

Guidelines:
- Answer in a conversational, helpful manner
- Strictly include inline citations: (Article Title, Link)
- If the question isn't overall health-related/ biomedical, politely redirect
- If insufficient information, say so clearly
- Use bullet points for symptoms, treatments, etc.
- Strictly use the provided context only. If no relevant information, say so.
- If context is not enough, say so.


Context from NHS articles:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        response = (prompt | llm).invoke({"context": context, "question": query})
        
        rephrased_info = f"🔄 **Rephrased for search:** {rephrased_query}" if was_rephrased and st.session_state.show_rephrased else None
        
        return response.content, rephrased_info, True
        
    except Exception as e:
        return f"Error: {e}", None, False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show rephrased query if it exists
        if "rephrased_info" in message and message["rephrased_info"]:
            st.info(message["rephrased_info"])

# Example questions (only show when no chat history)
# if not st.session_state.messages:
#     st.markdown("### 💡 Try asking:")
#     example_questions = [
#         "What are the symptoms of diabetes?",
#         "How can I manage anxiety?",
#         "What treatments are available for migraines?"
#     ]
    
#     cols = st.columns(len(example_questions))
#     for i, question in enumerate(example_questions):
#         with cols[i]:
#             if st.button(question, key=f"example_{i}"):
#                 # Add user message
#                 st.session_state.messages.append({"role": "user", "content": question})
                
#                 # Get response
#                 with st.chat_message("user"):
#                     st.markdown(question)
                
#                 with st.chat_message("assistant"):
#                     with st.spinner("Searching NHS knowledge base..."):
#                         response, rephrased_info, success = process_medical_query(question)
                    
#                     st.markdown(response)
#                     if rephrased_info:
#                         st.info(rephrased_info)
                
#                 # Add assistant response
#                 st.session_state.messages.append({
#                     "role": "assistant", 
#                     "content": response,
#                     "rephrased_info": rephrased_info
#                 })
#                 st.rerun()

# Chat input
if prompt := st.chat_input("Ask your medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching NHS knowledge base..."):
            response, rephrased_info, success = process_medical_query(prompt)
        
        st.markdown(response)
        if rephrased_info:
            st.info(rephrased_info)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "rephrased_info": rephrased_info
    })

# Footer
st.markdown("---")
st.caption("⚠️ This is for informational purposes only. Always consult healthcare professionals for medical advice.")