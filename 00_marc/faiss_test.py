import faiss
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

with torch.no_grad():
    model = HuggingFaceEmbeddings(model_name='sentence-transformers/LaBSE')
    a = model.embed_query("dummy")
    print(len(a))
