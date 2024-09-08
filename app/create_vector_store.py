from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.mongo_client import MongoClient
from langchain_openai import OpenAIEmbeddings
import urllib.parse
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_mongo_collection():
    load_dotenv()
    username = urllib.parse.quote_plus(os.getenv("MONGO_USERNAME"))
    password = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD"))
    uri = "mongodb+srv://" + username + ":" + password + "@cluster0.fsrq9cq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    mongo_client = MongoClient(uri)
    db = os.getenv("MONGO_DB_NAME")
    collection_name = os.getenv("MONGO_COLLECTION")

    collection = mongo_client[db][collection_name]
    return collection

def fetch_documents():

    opinions_directory = "opinions/"
    txt_files = [
        os.path.join(opinions_directory, file)
        for file in os.listdir(opinions_directory)
        if file.endswith(".txt")
    ]
    file_paths = txt_files
    docs = []
    for file in file_paths:
        print(f"Loading {file.title()}...")
        loader = TextLoader(file)
        doc = loader.load()[0]
        docs.append(doc)
    return docs

def split_docs():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    docs = fetch_documents()
    splits = text_splitter.split_documents(docs)
    return splits

def create_vector_store():

    collection = get_mongo_collection()
    splits = split_docs()
    try:
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=collection,

            index_name="opinion_index",
        )
    except Exception as e:
        print ("Error: ", e)
    
    return vector_store

create_vector_store()