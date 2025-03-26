import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings

class VectorDatabase:
    """Handles FAISS vector database operations."""
    
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings()
        self.texts = ["What is RAG?", "How does AWS Bedrock work?", "LangChain benefits."]
        self.index = self.create_index()

    def create_index(self):
        """Creates and returns a FAISS index."""
        embeddings = [self.embedding_model.embed_query(text) for text in self.texts]
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        return index

    def query(self, text):
        """Searches the FAISS index for the closest match."""
        query_vector = np.array([self.embedding_model.embed_query(text)], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k=1)
        return self.texts[indices[0][0]]
