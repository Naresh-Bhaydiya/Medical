# import os
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.vectorstores import Pinecone as PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferWindowMemory
# from flask import Flask, request, render_template
# from dotenv import load_dotenv

# load_dotenv()

# app = Flask(__name__)

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index_name = "geminivectorstore"

# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric='cosine',
#         spec=ServerlessSpec(cloud='aws', region='us-west-2')
#     )

# index = pc.__init__(index_name)
# embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

# memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", input_key="question")

# def get_conversational_chain():
#     prompt_template = """You are a Medical Assistant chatbot with expertise in healthcare and medical-related queries.

#     Always use the answers from vector storage (context) along with responses from Gemini-Pro to provide the best answers to the user.
#     If the user asks about medical advice, guidelines, or information, ensure the response is precise and contextually accurate. Do not provide casual or misleading responses.

#     Context:\n{context}\n
#     Question:\n{question}\n
#     Chat History:\n{chat_history}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, memory=memory)
#     return chain

# @app.route('/')
# def index_page():
#     return render_template('index.html')

# @app.route('/query', methods=['GET'])
# def query():
#     query = request.args.get('text')
#     vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#     document_context = vector_store.similarity_search(query, k=3)
#     chain = get_conversational_chain()
    
#     response = chain({"input_documents": document_context, "question": query}, return_only_outputs=True)
#     print("Response:", response)
#     return response["output_text"]

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)



import streamlit as st
from retrieve import MedicalVectorSearch

def main():
    # Set page configuration
    st.set_page_config(page_title="Medical Information Assistant", page_icon="🩺")

    # Initialize the medical vector search
    try:
        medical_bot = MedicalVectorSearch()
    except Exception as e:
        st.error(f"Error initializing Medical Vector Search: {e}")
        return

    # Title and description
    st.title("🩺 Medical Information Assistant")
    st.write("Get informed medical insights based on your health queries.")

    # Input area for medical queries
    query = st.text_input("Enter your medical question:", 
                           placeholder="What would you like to know about a medical condition?")

    # Sidebar for additional controls
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of similar context documents", 1, 5, 3)

    # Generate response when query is submitted
    if query:
        with st.spinner("Searching medical information and generating response..."):
            try:
                # Find similar texts with user-defined top_k
                medical_bot.top_k = top_k
                response = medical_bot.generate_response(query)

                # Display response
                st.subheader("Medical Insight")
                st.markdown(response)

                # Show similar context (optional)
                with st.expander("Similar Medical Contexts"):
                    similar_texts = medical_bot.find_similar_texts(query, top_k)
                    for i, match in enumerate(similar_texts, 1):
                        st.markdown(f"**Context {i} (Similarity: {match['score']:.2f})**")
                        st.text(match['text'])

            except Exception as e:
                st.error(f"Error generating medical response: {e}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.warning("Disclaimer: This is an AI assistant and does not replace professional medical advice. Always consult a healthcare professional.")

if __name__ == "__main__":
    main()