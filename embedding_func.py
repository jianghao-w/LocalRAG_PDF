from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
def get_emb_func():
    ollama_emb = OllamaEmbeddings(
        # model="llama3",
        model = "snowflake-arctic-embed"
    )
    return ollama_emb

# r1 = ollama_emb.embed_documents(
#     [
#         "Alpha is the first letter of Greek alphabet",
#         "Beta is the second letter of Greek alphabet",
#     ]
# )