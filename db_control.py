from langchain_community.document_loaders import PyPDFDirectoryLoader  # use Langchain pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # for slicing the pdf into small chunks
from langchain.schema.document import Document
from langchain_community.vectorstores.chroma import Chroma
from embedding_func import get_emb_func
from langchain_community.vectorstores import FAISS
import argparse
import os
import shutil
# import faiss

DATA_PATH = './test_data'
CHROMA_PATH = "chroma"

def main():

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    else:
        documents = load_pdf_docs()
        chunks = split_docs(documents)
        add_to_chroma(chunks)
        # add_to_faiss(chunks)


def add_to_faiss(chunks:list[Document]):
    db = FAISS.from_documents(
        chunks, get_emb_func()
    )
    query = "How to get out of jail in monopoly?"
    docs = db.similarity_search(query, k=5)
    print(docs)



def add_to_chroma(chunks:list[Document]):
    db = Chroma(
        persist_directory = CHROMA_PATH, embedding_function=get_emb_func()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def load_pdf_docs():
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    return pdf_loader.load()    # return list[Document]

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex = False,
    )

    return text_splitter.split_documents(documents)

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# docs = load_pdf_docs()
# print(docs[0])
# print(' ------------------------- ')
# chunks = split_docs(docs)
# print(chunks[0])


if __name__ == "__main__":
    main()