from langchain_community.document_loaders import TextLoader, WebBaseLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import urllib.parse
from openai import OpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi 
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory,ConversationBufferMemory






# Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)


class LangChain:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    

    def __init__(self):

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        self.uri=os.getenv('MONGO_URI')
        client = MongoClient(self.uri, server_api=ServerApi('1'))
        db = client[os.getenv('MONGO_CLIENT')] 
        
        self.collection = db[os.getenv('MONGO_COLLECTION')]
        if db:
            print("initialized mongo")

    def fetch_files_from_transcripts(self):
        """
        Fetches files from the transcripts directory and stores them in the 'file_paths' attribute.
        """
        # Define the directory containing .txt files
        transcripts_directory = "resources"

        # Get the list of .txt files in the directory
        txt_files = [
            os.path.join(transcripts_directory, file)
            for file in os.listdir(transcripts_directory)
            if file.endswith(".txt")
        ]

        self.file_paths = txt_files

    def text_based_loader(self):
        """
        Load, chunk and index the contents of the text file.
        """
        self.docs = []
        for file in self.file_paths:
            print(f"Loading {file}...")
            loader = TextLoader(file)
            doc = loader.load()[0]
            self.docs.append(doc)

    def text_splitter(self):
        """
        Split the blog into smaller chunks of text.
        """
        print("Splitting blog into smaller chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=500
        )
        self.splits = text_splitter.split_documents(self.docs)
        print("Splitted blog into smaller chunks...")

    
    def create_vector_store(self):
        
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        MongoDBAtlasVectorSearch.from_documents( self.splits , embeddings, collection=self.collection )

    
    def get_response(self):
        vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
            self.uri,
            "steinn_database" + "." + "vectorstore_clean_wo_book",
            OpenAIEmbeddings(disallowed_special=()),
            index_name= "vector_index_clean_wo_book",
        )
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=self.openai_api_key,
        )
        query = "why is 24 hour urine test not accurate?"
        chain = load_qa_chain(llm, chain_type="stuff")
        
        prompt_template = """
        You must only follow the instructions in list below:

        You are a friendly and conversational assistant named RAAFYA.
        Answer the questions based on the document or if the user asked something
        Text: {context}

        Question: {question}
        Answer :
        """
        

        while True:
            question = input("Ask me anything about the files (type 'exit' to quit): ")

            if question.lower() in ["exit"] and len(question) == 4:
                end_chat = "Thank you for visiting us! Have a nice day"
                print(end_chat)
                break

            if question != "":
                results = vectorStore.similarity_search(
                query=question, k=4
                )
                response = chain.run(input_documents=results, question=question)
                
                for result in results:
                    print('\nSOURCE')
                    print(result)
                print("--------------------------------")
                print(response)
                
                print("--------------------------------")
                

    def initialize_vector_store(self):
        """
        Set up the vector store, which will be used by the retriever.
        """
        print("Initializing vector store...")
        # self.fetch_files_from_transcripts()
        # self.text_based_loader()
        # self.text_splitter()
        # self.create_vector_store()
        # self.store_in_mongodb()
        self.get_response()
        
        
        
    


if __name__ == "__main__":
    langchain = LangChain()
    langchain.initialize_vector_store()
    