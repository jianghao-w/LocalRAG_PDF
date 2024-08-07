import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embedding_func import get_emb_func
import asyncio
import concurrent.futures

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# def main():
#     # Create CLI.
#     print('Welcome! Please type query:')
#     while (query_input := input('>>')) != 'exit':
#         query_rag(query_input)
async def handle_query(loop, pool, query_input):
    if query_input == 'exit':
        return
    # Perform the query (you might need to adjust this to fit your actual model and query function)
    result = await loop.run_in_executor(pool, query_rag, query_input)
    # print(result)

async def main():
    with concurrent.futures.ThreadPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        while True:
            query_input = input('>>')
            if query_input == 'exit':
                break
            await handle_query(loop,pool, query_input)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_emb_func()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=6)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print('\nprompt:\n',prompt)

    model = Ollama(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    asyncio.run(main())