import os
from dotenv import load_dotenv

# Document Loading
from langchain_community.document_loaders import PyPDFLoader
# Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# Vector Store
from langchain_community.vectorstores import Chroma

# Define the path for the persistent vector store
PERSIST_DIRECTORY = "./chroma_db"

def build_rag_pipeline():
    """
    Builds and tests the RAG pipeline by loading a PDF, splitting it,
    creating embeddings, and storing them in a Chroma vector store.
    """
    print("--- Starting RAG Pipeline Build ---")
    load_dotenv()

    # 1. Load Documents
    print("\n--- Loading Documents ---")
    # For this test, we'll load one document.
    # The path is relative to the project root.
    document_path = "data/my_document.pdf"
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"PDF document not found at {document_path}. Please add it.")

    loader = PyPDFLoader(document_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from {document_path}.")
    print(f"Content of first page: {documents[0].page_content[:200]}...")

    # 2. Split Documents into Chunks
    print("\n--- Splitting Documents ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    print(f"Content of first chunk: {chunks[0].page_content}")

    # 3. Create Embeddings
    print("\n--- Creating Embeddings ---")
    # We will use a powerful, open-source embedding model from Hugging Face.
    # This will be downloaded automatically the first time you run it.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store Chunks in Chroma Vector Store
    print("\n--- Storing Chunks in Chroma ---")
    # This will create a persistent vector store in the 'chroma_db' directory.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"Successfully created and persisted vector store with {vector_store._collection.count()} vectors.")

    # 5. Test the Retriever
    print("\n--- Testing the Retriever ---")
    query = "What is the main topic of this document?"
    retrieved_docs = vector_store.similarity_search(query, k=3) # Get top 3 most similar chunks

    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(retrieved_docs)} documents.")
    print("\n--- Top Retrieved Document ---")
    print(retrieved_docs[0].page_content)

    print("\n--- RAG Pipeline Build Finished ---")

if __name__ == "__main__":
    build_rag_pipeline()