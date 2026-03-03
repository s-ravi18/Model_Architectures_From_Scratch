from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "embeddings/chroma"


def get_retriever():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )