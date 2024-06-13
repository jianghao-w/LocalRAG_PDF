from langchain.document_loaders.pdf import PyPDFDirectoryLoader  # use Langchain pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # for slicing the pdf into small chunks
from langchain.schema.document import Document
PATH = './test_data'

def load_pdf_docs():
    pdf_loader = PyPDFDirectoryLoader(PATH)
    return pdf_loader.load()

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 768,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )

    return text_splitter.split_documents(documents)


docs = load_pdf_docs()
print(docs[0])
print(' ------------------------- ')
chunks = split_docs(docs)
print(chunks[0])