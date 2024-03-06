from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM
import torch
from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import SummaryIndex
# from llama_index.core.response.notebook_utils import display_response
from llama_index.core import StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
import time

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device= device)

hf_token = "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"

Settings.llm = Ollama(model="llama2", request_timeout=60.0)
# llm = LlamaCPP(
#     model_path= "llama-2-13b-chat.Q4_0.gguf",
#     temperature= 0.1,
#     max_new_tokens= 256,
#     context_window= 3900,
#     generate_kwargs= {},
#     model_kwargs= {"n_gpu_layers": 100},
#     verbose= True,
#     messages_to_prompt= messages_to_prompt,
#     completion_to_prompt= completion_to_prompt,
# )

loader = PDFReader()
# file_name = input("Input your file name with the path")
documents = loader.load_data(file=Path("The McKinsey Way.pdf"))

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection("Mckinsey_Way")
vector_store = ChromaVectorStore(chroma_collection= collection)
index = VectorStoreIndex.from_vector_store(vector_store = vector_store, embed_model = embed_model)

start = time.time()
query_engine = index.as_query_engine()
response = query_engine.query("What is the book about?")
print(response.response)
print(time.time() - start)

# vector_index = VectorStoreIndex.from_documents(documents= documents, embed_model = embed_model, llm = llm, device = device)

# summary_index = SummaryIndex.from_documents(documents= documents, embed_model = embed_model, llm = llm, device = device)

# question = input("Ask any question about the document")
# while(True):
#     query_engine = vector_index.as_query_engine(response_mode = "compact", llm= llm)
#     response = query_engine.query(question)
#     print(response.response)


# query_engine = vector_index.as_query_engine(response_mode = "compact", llm= llm)
# response = query_engine.query("Can you summarize the key aspects of what the book covers?")
# display_response(response)

# print(response.response)