import os
import boto3
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import Bedrock

# Initialize FastAPI
app = FastAPI()

# AWS Bedrock Client Setup
boto3_session = boto3.Session(region_name="us-east-1")
bedrock_client = boto3_session.client(service_name="bedrock-runtime")

# FAISS Vector Database Setup
def create_faiss_index():
    """Creates a FAISS vector database."""
    embedding_model = OpenAIEmbeddings()
    texts = ["Explain RAG", "How does AWS Bedrock work?", "Use cases of LangChain."]
    embeddings = [embedding_model.embed_query(text) for text in texts]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, texts

faiss_index, stored_texts = create_faiss_index()

@app.get("/query")
def query_rag(input_text: str):
    """Queries AWS Bedrock for an AI-generated response."""
    try:
        embedding_model = OpenAIEmbeddings()
        query_vector = np.array([embedding_model.embed_query(input_text)], dtype=np.float32)
        distances, indices = faiss_index.search(query_vector, k=1)
        matched_text = stored_texts[indices[0][0]]
        
        # Query AWS Bedrock
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body={"prompt": f"Answer this query: {matched_text}"}
        )
        return {"query": input_text, "response": response["body"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
