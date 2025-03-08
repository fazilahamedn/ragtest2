import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define path for embeddings in root directory
EMBEDDINGS_PATH = "faiss_index"


def create_embeddings():
    """Create and save embeddings using HuggingFace model"""
    print("Initializing embedding model...")

    # Initialize embeddings model with explicit model kwargs
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Force CPU usage to avoid CUDA issues
    )

    print("Creating FAISS vector store...")

    # Create a placeholder document since FAISS needs at least one document to initialize
    # In a real application, you would add your actual documents here
    placeholder_doc = Document(page_content="This is a placeholder document for initializing the vector store.")

    # Create FAISS index with the placeholder document
    vectors = FAISS.from_documents([placeholder_doc], embeddings)

    print(f"Saving FAISS index to {EMBEDDINGS_PATH}...")
    vectors.save_local(EMBEDDINGS_PATH)

    print("Embeddings created and saved successfully!")
    return vectors


if __name__ == "__main__":
    print("Starting embeddings creation process...")
    try:
        create_embeddings()
        print("Process completed successfully!")
    except Exception as e:
        print(f"Error during embeddings creation: {e}")
        sys.exit(1)