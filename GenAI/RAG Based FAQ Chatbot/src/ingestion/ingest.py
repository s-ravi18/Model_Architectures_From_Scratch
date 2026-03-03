import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "embeddings/chroma"
FAQ_PATH = "data/faqs.json"


def load_faq_documents():
    with open(FAQ_PATH, "r") as f:
        data = json.load(f)

    documents = []
    for faq in data["faqs"]:
        text = (
            f"Question: {faq['question']}\n"
            f"Answer: {faq['answer']}\n"
            f"Category: {faq['category']}"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "faq_id": faq["id"],
                    "category": faq["category"]
                }
            )
        )
    return documents


def ingest():
    docs = load_faq_documents()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectordb.persist()
    print("✅ FAQ ingestion completed")


if __name__ == "__main__":
    ingest()