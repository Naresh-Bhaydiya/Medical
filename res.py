import os
import streamlit as st
from decouple import config
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from cohere import Client


class MedicalVectorSearch:
    def __init__(self):
        try:
            # Load API keys from environment variables
            self.COHERE_API_KEY = config("COHERE_API_KEY")
            self.PINECONE_API_KEY = config("PINECONE_API_KEY")
            self.PINECONE_ENVIRONMENT = config("PINECONE_ENVIRONMENT", default="us-east-1")

            # Initialize Pinecone client
            self.pinecone_client = Pinecone(api_key=self.PINECONE_API_KEY, environment=self.PINECONE_ENVIRONMENT)

            # Initialize Cohere embeddings and client
            self.embeddings = CohereEmbeddings(
                cohere_api_key=self.COHERE_API_KEY, 
                model="embed-english-v2.0"
            )
            self.cohere_client = Client(api_key=self.COHERE_API_KEY)

            # Define index name
            self.INDEX_NAME = "pdf-embeddings"

            # Create index if it doesn't exist
            self._create_index_if_not_exists()

            # Connect to the index
            self.index = self.pinecone_client.Index(self.INDEX_NAME)
        except Exception as e:
            raise RuntimeError(f"Error during initialization: {e}")
            
    def _create_index_if_not_exists(self):
        try:
            indexes = self.pinecone_client.list_indexes()
            if self.INDEX_NAME not in indexes:
                print(f"Creating index: {self.INDEX_NAME}")
                self.pinecone_client.create_index(
                    name=self.INDEX_NAME,
                    dimension=4096,  # Ensure this matches your embedding model
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.PINECONE_ENVIRONMENT),
                )
            else:
                print(f"Index {self.INDEX_NAME} already exists.")
        except Exception as e:
            raise RuntimeError(f"Error creating index: {e}")

    def upsert_documents(self, documents):
        try:
            vectors = [
                (
                    str(doc["id"]),
                    self.embeddings.embed_query(doc["text"]),
                    {"text": doc["text"]}
                ) for doc in documents
            ]
            self.index.upsert(vectors)
        except Exception as e:
            raise RuntimeError(f"Error upserting documents: {e}")

    def find_similar_texts(self, input_text, top_k=3):
        try:
            input_embedding = self.embeddings.embed_query(input_text)
            result = self.index.query(vector=input_embedding, top_k=top_k, include_metadata=True)
            return [
                {"text": match["metadata"]["text"], "score": match["score"]}
                for match in result["matches"]
            ]
        except Exception as e:
            raise RuntimeError(f"Error finding similar texts: {e}")

    def generate_response(self, input_text, top_k=3):
        try:
            similar_texts = self.find_similar_texts(input_text, top_k)
            context = " ".join([match["text"] for match in similar_texts])
            response = self.cohere_client.generate(
                prompt=f"""
                Medical Consultation Context:
                User Query: {input_text}

                Relevant Medical Information:
                {context}

                Based on the context and query, provide a clear, informative, and empathetic medical explanation:
                """,
                max_tokens=500,
                temperature=0.3
            )
            return response.generations[0].text
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")


def streamlit_app():
    st.set_page_config(page_title="Medical Assistant", page_icon="🩺")
    st.title("🩺 Medical Information Assistant")
    st.write("Get informed medical insights based on your queries.")

    try:
        bot = MedicalVectorSearch()
    except RuntimeError as e:
        st.error(f"Initialization failed: {e}")
        return

    query = st.text_input("Enter your medical question:", placeholder="e.g., What is hypertrophic cardiomyopathy?")
    top_k = st.sidebar.slider("Number of similar context documents", 1, 5, 3)

    if query:
        with st.spinner("Processing your request..."):
            try:
                response = bot.generate_response(query, top_k)
                st.subheader("Medical Insight")
                st.markdown(response)

                with st.expander("Similar Medical Contexts"):
                    similar_texts = bot.find_similar_texts(query, top_k)
                    for i, match in enumerate(similar_texts, 1):
                        st.markdown(f"**Context {i} (Score: {match['score']:.2f})**")
                        st.text(match["text"])
            except RuntimeError as e:
                st.error(f"Error generating response: {e}")

    st.sidebar.warning("Disclaimer: This is an AI assistant and does not replace professional medical advice.")


if __name__ == "__main__":
    streamlit_app()
