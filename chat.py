from flask import Flask, render_template, request, session

import os
from dotenv import load_dotenv
import urllib.parse
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv("SECRET_KEY")


conversation_history = []


class LangChain:
    """
    The LangChain class is responsible for loading the relevant blog,
    processing it, and then generating answers using OpenAI's ChatOpenAI.
    """

    def __init__(self):
        self.template = """You are assisting a hormone therapy doctor. Use the following pieces of transcript to answer the question at the end.
        If you don't know the answer or if the answer does not belong in the context, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        """
        self.store = {}
        username = urllib.parse.quote_plus(os.getenv("MONGO_USERNAME"))
        password = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD"))
        self.uri = "mongodb+srv://" + username + ":" + password + "@cluster0.fsrq9cq.mongodb.net/"

    # vector store code starts here

    def fetch_files_from_transcripts(self):
        """
        Fetches files from the transcripts directory and stores them in the 'files' attribute.
        No parameters or return types specified.
        """

        # Define the directory containing .txt files
        transcripts_directory = "transcripts"

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
            print(f"Loading {file.title()}...")
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

    def vector_store(self):
        """
        Create a vector store from the smaller chunks of text.
        """
        print("Connecting vector store...")

        client = MongoClient(self.uri, server_api=ServerApi("1"))
        db = client[os.getenv("MONGO_DB_NAME")]
        collection = db[os.getenv("MONGO_COLLECTION")]

        embeddings = OpenAIEmbeddings()

        MongoDBAtlasVectorSearch.from_documents(
            self.splits, embeddings, collection=collection
        )

    def initialize_vector_store(self):
        """
        Set up the vector store, which will be used by the retriever.
        """
        print("Initializing vector store...")
        # self.web_based_loader()
        # self.text_based_loader()
        # self.text_splitter()
        self.vector_store()

    # vectorstore code ends here

    def initialize_retriever(self):
        """
        Set up the retriever, which will retrieve the relevant snippets of text from the blog.
        """
        print("Initializing retriever...")

        vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            self.uri,
            "steinn_database.vectorstore_atlas_1",
            OpenAIEmbeddings(disallowed_special=()),
            index_name="vector_index_1",
        )
        self.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 1}
        )
        self.openai_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def initialize_prompt(self):
        contextualize_q_system_prompt = """Given a transcript, chat history and latest user question \
        which might reference context in the chat history. Keep the context as it is. DO NOT answer the question, \
        just reformulate the question if needed and otherwise return it as is.
        """
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

    def select_model(self, llm_model):
        history_aware_retriever = create_history_aware_retriever(
            llm_model, self.retriever, self.contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(
            llm_model, self.chat_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        # if we want to keep context to a certain limit, we can add trim_messages function in front of the chain
        self.chain_with_message_history = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    # def pretty_print_docs(docs):
    #     print(
    #         f"\n{'-' * 100}\n".join(
    #             [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
    #         )
    #     )
    def get_relevant_docs(self):
        vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            self.uri,
            "steinn_database.vectorstore_atlas_1",
            OpenAIEmbeddings(disallowed_special=()),
            index_name="vector_index_1",
        )
        self.relevant_docs = vectorstore.similarity_search(query="query", k=1)


langchain = LangChain()
print("started fetching")
while True:
    # langchain.initialize_vector_store()
    # langchain.initialize_retriever()
    # langchain.initialize_prompt()
    # langchain.select_model(langchain.openai_llm)
    # break
    langchain.get_relevant_docs()
    break
print("done fetching")


@app.route("/")
def home():
    if "logged_in" in session and session["logged_in"]:
        global conversation_history
        conversation_history = []
        return render_template("chat.html")
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login():
    password = request.form["password"]
    if password == os.getenv("SESSION_PASSWORD"):
        session["logged_in"] = True
        global conversation_history
        conversation_history = []

        return render_template("chat.html", history=conversation_history)


# Define route for processing form submission
@app.route("/chat", methods=["POST", "GET"])
def process_form():
    if session.get("session_id") is None:
        session_id = os.urandom(2)
    else:
        session_id = session.get("session_id")

    user_question = session.get("question")
    # langchain.history.add_user_message(user_question)
    conversation_history.append(("User", user_question))
    response = langchain.chain_with_message_history.invoke(
        # {"context": langchain.relevant_docs[0].page_content},
        {"input": user_question},
        {"configurable": {"session_id": session_id}},
    )
    conversation_history.append(("Model", response["answer"]))
    # conversation_history.append(('Model', full_response))
    full_response = "\n".join(
        [str(response["answer"]), "Source:", str(response["context"])]
    )

    print(response)
    return render_template("chat.html", history=conversation_history)


@app.route("/new")
def reset():
    global conversation_history
    conversation_history = []
    session_id = os.urandom(2)
    session["session_id"] = session_id
    langchain.chain_with_message_history.invoke(
        # {"context": langchain.relevant_docs[0].page_content},
        {"input": ""},
        {"configurable": {"session_id": session_id}},
    )["answer"]
    return render_template("chat.html", history=conversation_history)


@app.route("/error")
def error():
    return render_template("error.html")


@app.route("/load", methods=["POST"])
def show_chat():
    session["question"] = request.form["question"]
    return render_template("loading.html", history=conversation_history)


if __name__ == "__main__":
    app.run(debug=True)
