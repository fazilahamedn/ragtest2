import streamlit as st
import os
import re
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use the same directory path calculation as backend.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_NAME = "faiss_index"
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, INDEX_NAME)

# Check for Groq API key
groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    st.error("Groq API key not found. Please check your .env file.")


def load_embeddings():
    """Load existing embeddings from the current directory"""
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            st.info(f"Loading existing embeddings from {EMBEDDINGS_PATH}...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}  # Force CPU usage to avoid CUDA issues
            )
            vectors = FAISS.load_local(EMBEDDINGS_PATH, embeddings, allow_dangerous_deserialization=True)
            st.success("Embeddings loaded successfully!")
            return vectors
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return None
    else:
        st.error(f"Embeddings not found at {EMBEDDINGS_PATH}. Please run backend.py first to create embeddings.")
        return None


# Initialize session state variables
if "vectors" not in st.session_state:
    st.session_state.vectors = load_embeddings()

if "submitted" not in st.session_state:
    st.session_state.submitted = False


# Function to handle form submission
def handle_submit():
    st.session_state.submitted = True


# Page title
st.markdown("<h1 style='text-align: center;'>RAG Demo</h1>", unsafe_allow_html=True)

# Only proceed if embeddings are loaded
if st.session_state.vectors:
    try:
        # Initialize the LLM
        llm = ChatGroq(model="deepseek-r1-distill-qwen-32b")

        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Create the retriever
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Create a form for user input
        with st.form(key="query_form"):
            user_prompt = st.text_input("Input your prompt here", key="user_input", max_chars=200,
                                        label_visibility="collapsed")
            submit_button = st.form_submit_button("Submit", on_click=handle_submit)

        # Handle the submission
        if st.session_state.submitted:
            if user_prompt:  # Only process if there's actual input
                with st.spinner("Generating response..."):
                    start = time.time()
                    try:
                        response = retrieval_chain.invoke({"input": user_prompt})
                        end = time.time()

                        # Remove the <think>...</think> section from the answer
                        cleaned_answer = re.sub(r'<think>.*?</think>', '', response['answer'], flags=re.DOTALL).strip()

                        # Display the answer
                        st.write(cleaned_answer)
                        st.caption(f"Response time: {end - start:.2f} seconds")

                        # Show retrieved documents
                        with st.expander("Document Similarity Search"):
                            for doc in response.get("context", []):
                                st.write(doc.page_content)
                                st.write("--------------------------------")
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

            # Reset the submitted flag for the next interaction
            st.session_state.submitted = False
    except Exception as e:
        st.error(f"Error initializing LLM or chain: {e}")
else:
    st.warning("Please run backend.py first to create embeddings before using this interface.")