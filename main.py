import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Custom Embedding Function Wrapper
class CohereEmbeddingFunction:
    def __init__(self, cohere_client):
        self.client = cohere_client
    
    def embed_documents(self, texts):
        response = self.client.embed(texts=texts, model="embed-english-v2.0")
        return response.embeddings
    
    def embed_query(self, text):
        response = self.client.embed(texts=[text], model="embed-english-v2.0")
        return response.embeddings[0]

# Initialize Flask app
app = Flask(__name__)

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# Initialize Pinecone client 
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdf-embeddings"

# Check if index exists, otherwise create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )

# Access the index 
index = pc.Index(index_name)

# Initialize vector store with updated method
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=CohereEmbeddingFunction(co)
)

# Initialize memory for conversation
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", input_key="question")

# Setup QA Chain
def get_conversational_chain():
    prompt_template = """
    You are a Medical Assistant chatbot with expertise in healthcare and medical-related queries.

    Use the context from the vector store along with Cohere's generation model to provide answers.
    Ensure the response is accurate and detailed when necessary.

    Context:
    {context}

    Question:
    {question}

    Chat History:
    {chat_history}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"],
    )
    return load_qa_chain(
        llm=co, 
        chain_type="stuff", 
        prompt=prompt, 
        memory=memory
    )


@app.route("/")
def index():
    return render_template("index.html")

from retrieve import MedicalVectorSearch

# @app.route("/query", methods=["GET"])
# def query():
#     user_query = request.args.get("text", "")
#     try:
#         # Search similar documents in Pinecone

#         medical_bot = MedicalVectorSearch()
#         response = medical_bot.generate_response(user_query)
#         # # Run QA chain with the context
#         # chain = get_conversational_chain()
#         # answer = chain.run(input={"context": context_docs, "question": user_query})
#         return jsonify({"response": response})
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         return jsonify({"error": str(e)}), 500



@app.route("/query", methods=["GET"])
def query():
    user_query = request.args.get("text", "")
    try:
        # Initialize your bot logic here (replace with your actual implementation)
        medical_bot = MedicalVectorSearch()
        response = medical_bot.generate_response(user_query)

        # Respond with the generated answer
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)