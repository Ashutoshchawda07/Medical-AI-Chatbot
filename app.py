from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)  

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

print("--- 1. Keys Loaded ---") 

# --- START OF POTENTIAL HANG AREA ---
try:
    print("--- 2a. Starting embeddings download... ---")
    embeddings=download_embeddings()
    print("--- 2b. Embeddings downloaded successfully. ---")
    
    index_name = "medical-ai-chatbot"
    print(f"--- 3a. Connecting to Pinecone index: {index_name} ---")
    docsearch_store = PineconeVectorStore.from_existing_index(
       index_name=index_name,
       embedding=embeddings
    )
    print("--- 3b. Pinecone connection successful. ---")
except Exception as e:
    print(f"--- FATAL SETUP ERROR: {e} ---")
    exit() # Exit the application if setup fails
# --- END OF POTENTIAL HANG AREA ---


retriever = docsearch_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatmodel = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GEMINI_API_KEY  # <--- Uses the explicitly loaded key
)

prompt=ChatPromptTemplate.from_messages(
   [
      ("system", system_prompt),
      ("user", "{input}")
   ]
)

question_answer_chain=create_stuff_documents_chain(chatmodel,prompt=prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

print("--- 4. RAG Chain setup complete. App starting. ---")


@app.route('/')
def home():
  return render_template('index.html')

@app.route('/get', methods=['GET','POST'])
def ask():
    # --- FIX: Safely retrieve the 'message' key from either JSON or Form data ---
    if request.is_json:
        # If client sends JSON (standard for modern AJAX)
        data = request.json
    else:
        # If client sends form data (standard for HTML forms)
        data = request.form

    try:
        msg = data['message']
    except KeyError:
        # Handle the case where the key is still missing
        print("--- ERROR: 'message' key not found in request data. ---")
        return jsonify({"answer": "Error: Invalid request format. 'message' key missing."}), 400
    # --- END FIX ---
    
    input=msg
    print(f"--- 5. Received query: {input} ---")
    
    # --- START OF POTENTIAL TIMEOUT AREA ---
    try:
        response=rag_chain.invoke({"input":msg})
        print(f"--- 6. Response received successfully. ---")
        # Use jsonify for a proper API response
        return jsonify({"answer": response['answer']}) 
    except Exception as e:
        # Log the error if the RAG chain fails to invoke (e.g., API key problem, rate limit, timeout)
        print(f"--- 6. ERROR: RAG Chain failed to invoke: {e} ---")
        return jsonify({"answer": f"Error: Could not generate a response. Details: {e}"}), 500
    # --- END OF POTENTIAL TIMEOUT AREA ---


if __name__ == '__main__':
  app.run(host="0.0.0.0", port=5000, debug=True)
