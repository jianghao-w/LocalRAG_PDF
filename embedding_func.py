from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
def get_emb_func():
    ollama_emb = OllamaEmbeddings(
        # model="llama3",
        model = "snowflake-arctic-embed"
    )
    return ollama_emb