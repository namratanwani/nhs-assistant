import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.document import Document
import warnings

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv(override=True)

# Define constants
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
INDEX_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize LLM and Embeddings 
# Explicitly get the API key from environment and pass it to the LLM
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # Fallback to checking OPENAI_KEY as in your original code
    openai_api_key = os.getenv("OPENAI_KEY")
    
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY or OPENAI_KEY environment variable.")

# Initialize the LLM with the explicit API key
llm = ChatOpenAI(
    model="gpt-4.1", 
    temperature=0.2,
    openai_api_key=openai_api_key  # Explicitly pass the API key
)

# Initialize embeddings with caching for better performance
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    cache_folder="./embeddings_cache"  # Cache embeddings for better performance
)

# Check if index exists, otherwise create it ===
def create_or_load_vectorstore():
    if os.path.exists(INDEX_PATH) and os.path.isdir(INDEX_PATH) and len(os.listdir(INDEX_PATH)) > 0:
        print("Loading existing vector store...")
        vectorstore = Chroma(persist_directory=INDEX_PATH, embedding_function=embeddings)
    else:
        print("Building new vector store...")
        
        # Create directory if it doesn't exist
        os.makedirs(INDEX_PATH, exist_ok=True)
        
        # Load data
        df = pd.read_csv("data/nhs_articles.csv")
        
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
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=INDEX_PATH
        )
        
        # Save the vector store
        vectorstore.persist()
        print("Vector store built and saved.")
    
    return vectorstore

# Create the vector store and retriever ===
vectorstore = create_or_load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

#  Define the prompt template with your custom template 
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

# === STEP 5: Define the RAG chain ===
def format_docs(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        title = metadata.get("title", "Unknown")
        link = metadata.get("link", "Unknown")
        
        context += f"\nRelevant Article {i}:\nTitle: {title}\nLink: {link}\n{doc.page_content}\n\n"
    
    return context

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# === STEP 6: Create a convenient query function ===
def answer_medical_query(query):
    try:
        response = rag_chain.invoke(query)
        return response.content
    except Exception as e:
        return f"Error: {e}\nCheck your dependencies and OpenAI API key."

# Example usage
if __name__ == "__main__":
    print("\n=== Medical RAG System ===")
    print("\nQuery: What is PCOS and how does it affect fertility?")
    print("\nResponse:")
    print(answer_medical_query("What is PCOS and how does it affect fertility?"))