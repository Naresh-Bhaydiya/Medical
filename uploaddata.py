import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, Index, ServerlessSpec
from decouple import config

# Load API keys from environment variables
COHERE_API_KEY = config("COHERE_API_KEY")
PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = config("PINECONE_ENVIRONMENT")

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone index configuration
INDEX_NAME = "pdf-embeddings"
if INDEX_NAME not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=4096,  # Adjust dimension as needed
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

# Access the index
index = Index(
    name=INDEX_NAME,
    api_key=PINECONE_API_KEY,
    host=f"{PINECONE_ENVIRONMENT}"
)

# Step 1: Load and extract text from the PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Step 2: Chunk the data
def chunk_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    # print("================",chunks)
    return chunks

    
# Step 3: Embed chunks and store in Pinecone
def embed_and_store(chunks):
    # Initialize Cohere embeddings with a specified model
    embeddings = CohereEmbeddings(
        cohere_api_key=COHERE_API_KEY, 
        model="embed-english-v2.0"  # Replace with the model you prefer
    )

    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        metadata = chunk.metadata

        # Generate embeddings
        embedding = embeddings.embed_query(text)

        # Store in Pinecone
        index.upsert([
            {"id": f"doc-{i}", "values": embedding, "metadata": metadata}
        ])
        print(f"Chunk {i} stored successfully.")


# Main function
if __name__ == "__main__":
    # Specify your PDF file path
    pdf_file_path = "EKGBook.pdf"
    
    print("Loading PDF...")
    documents = load_pdf(pdf_file_path)
    
    print("Chunking text...")
    chunks = chunk_text(documents)
    
    print("Embedding and storing data in Pinecone...")
    embed_and_store(chunks)
    
    print("Process completed!")
