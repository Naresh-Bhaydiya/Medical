# from langchain_cohere import CohereEmbeddings
# from pinecone import Pinecone, Index, ServerlessSpec
# from decouple import config

# # Load API keys from environment variables
# COHERE_API_KEY = config("COHERE_API_KEY")
# PINECONE_API_KEY = config("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = config("PINECONE_ENVIRONMENT")

# # Initialize Pinecone client
# pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# # Define index name
# INDEX_NAME = "pdf-embeddings"

# # Check if the index exists; if not, create it
# if INDEX_NAME not in pinecone_client.list_indexes().names():
#     pinecone_client.create_index(
#         name=INDEX_NAME,
#         dimension=768,  # Adjust dimension based on your embedding model
#         metric="cosine",  # Use "cosine" or other metrics based on your needs
#         spec=ServerlessSpec(
#             cloud="aws",  # Specify your cloud provider
#             region=PINECONE_ENVIRONMENT,  # Specify your Pinecone environment region
#         )
#     )

# # Connect to the index
# index = Index(
#     name=INDEX_NAME,
#     api_key=PINECONE_API_KEY,
#     host=f"{PINECONE_ENVIRONMENT}"
# )

# # Initialize Cohere embeddings
# embeddings = CohereEmbeddings(
#     cohere_api_key=COHERE_API_KEY,
#     model="embed-english-v2.0",  # Specify the embedding model
#     user_agent="embed-english-v2.0"
# )

# def upsert_texts_to_pinecone(texts):
#     # Generate embeddings for texts
#     for i, text in enumerate(texts):
#         embedding = embeddings.embed_query(text)
        
#         # Upsert embedding into Pinecone
#         index.upsert([
#             {"id": f"text-{i}", "values": embedding, "metadata": {"text": text}}
#         ])
#         print(f"Text {i} upserted successfully.")

# def find_similar_texts(input_text, top_k=3):
#     # Generate embedding for the input text
#     input_embedding = embeddings.embed_query(input_text)

#     # Query Pinecone for similar embeddings
#     result = index.query(
#         vector=input_embedding,
#         top_k=top_k,
#         include_metadata=True  # Include original text metadata in the result
#     )

#     # Print similar texts
#     print("Top Similar Texts:")
#     for match in result["matches"]:
#         print(f"Similarity Score: {match['score']}")
#         print(f"Text: {match['metadata']['text']}")
#         print("-" * 50)

# if __name__ == "__main__":
#     # Example texts to store in Pinecone
#     texts = [
#         "This is a sample text about machine learning.",
#         "Another example about artificial intelligence.",
#         "A tutorial on embedding similarity and NLP."
#     ]

#     # Upsert texts into Pinecone
#     upsert_texts_to_pinecone(texts)

#     # Input text to find similar texts
#     input_text = "what is desease?"

#     # Find similar texts
#     find_similar_texts(input_text)


















import os
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from cohere import Client
from decouple import config

# Load API keys from environment variables
COHERE_API_KEY = config("COHERE_API_KEY")
PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = config("PINECONE_ENVIRONMENT", default="us-east-1")

class MedicalVectorSearch:
    def __init__(self):
        try:
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

            # Initialize Cohere embeddings and client
            self.embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v2.0")
            self.cohere_client = Client(api_key=COHERE_API_KEY)

            # Define index name
            self.INDEX_NAME = "pdf-embeddings"

            # Create index if it doesn't exist
            self._create_index_if_not_exists()

            # Connect to the index
            self.index = self.pinecone_client.Index(self.INDEX_NAME)

        except Exception as e:
            print(f"Error during initialization: {e}")

    def _create_index_if_not_exists(self):
        try:
            indexes = self.pinecone_client.list_indexes()
            if self.INDEX_NAME not in indexes:
                print(f"Creating index {self.INDEX_NAME}...")
                self.pinecone_client.create_index(
                    name=self.INDEX_NAME,
                    dimension=4096,  # Cohere embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"Index {self.INDEX_NAME} created successfully.")
            else:
                print(f"Index {self.INDEX_NAME} already exists.")
        except Exception as e:
            print(f"Error creating index: {e}")

    def upsert_documents(self, documents):
        try:
            # Prepare vectors for upserting
            vectors = [
                (
                    str(doc['id']),
                    self.embeddings.embed_query(doc['text']),
                    {"text": doc['text']}
                ) for doc in documents
            ]

            # Upsert vectors
            self.index.upsert(vectors)
            print(f"Upserted {len(documents)} documents.")
        except Exception as e:
            print(f"Error upserting documents: {e}")

    def find_similar_texts(self, input_text, top_k=3):
        try:
            # Generate embedding for input text
            input_embedding = self.embeddings.embed_query(input_text)

            # Query Pinecone for similar embeddings
            result = self.index.query(
                vector=input_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Extract and return similar texts
            matches = [
                {"text": match["metadata"]["text"], "score": match["score"]}
                for match in result["matches"]
            ]
            return matches
        except Exception as e:
            print(f"Error finding similar texts: {e}")
            return []

    def generate_response(self, input_text):
        try:
            # Find similar medical texts
            similar_texts = self.find_similar_texts(input_text)

            # Combine similar texts as context
            context = " ".join([match["text"] for match in similar_texts])

            # Generate response using Cohere
            response = self.cohere_client.generate(
                prompt=f"""You are a sophisticated Medical Consulting Bot designed to provide empathetic, accurate, and personalized medical guidance to patients. Your primary objectives are:
                1. Consultation Purpose:
                - Offer reliable medical information and advice
                - Provide compassionate and clear explanations
                - Support patients in understanding their health queries
                - Retrieve and synthesize relevant medical information from a comprehensive vector store knowledge base

                2. Interaction Guidelines:
                - Maintain a professional yet warm and supportive tone
                - Adapt communication style to patient's query complexity
                - Prioritize patient understanding and comfort
                - Ensure responses are clear, precise, and evidence-based
                - Respect patient privacy and confidentiality

                3. Response Framework:
                - Carefully analyze the user's input to understand context and intent
                - Cross-reference user query with extensive medical knowledge database
                - Generate responses that are:
                * Medically accurate
                * Easily comprehensible
                * Tailored to the patient's specific needs
                * Empathetic and non-intimidating

                4. Scope of Assistance:
                - Handle medical information queries
                - Provide general health advice
                - Offer preliminary guidance on symptoms
                - Engage in casual conversation when appropriate
                - Clarify medical terminologies
                - Suggest potential next steps for healthcare

                5. Important Limitations:
                - Clearly state that you are an AI assistant, not a replacement for professional medical diagnosis
                - Recommend consulting healthcare professionals for definitive medical advice
                - Avoid providing diagnosis
                - Do not prescribe medications
                - Emphasize the importance of professional medical consultation for serious concerns

                6. Contextual Response Strategy:
                - Utilize retrieved similar texts from vector store as contextual background
                - Synthesize information to create a coherent, informative response
                - Ensure responses are relevant, precise, and aligned with retrieved medical knowledge

                Response Tone: Professional, empathetic, clear, and supportive
                User Query: {input_text}

                Relevant Medical Information:
                {context}

                Based on the context and query, provide a clear, informative, and empathetic medical explanation:""",
                max_tokens=500,
                temperature=0.3
            )

            return response.generations[0].text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response at this time."


