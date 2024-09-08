

import os
from dotenv import load_dotenv
import urllib.parse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_mongodb import MongoDBAtlasVectorSearch



load_dotenv()


conversation_history = []

class LangChain:
    """
    The LangChain class is responsible for loading the relevant blog,
    processing it, and then generating answers using OpenAI's ChatOpenAI.
    """

    def __init__(self):
        self.template = """You are a human replica. Use the following pieces of opinions and views to answer the question at the end.
        If you don't know the answer or if the answer does not belong in the context, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        """
        self.store = {}
        username = urllib.parse.quote_plus(os.getenv("MONGO_USERNAME"))
        password = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD"))
        self.uri = "mongodb+srv://" + username + ":" + password + "@cluster0.fsrq9cq.mongodb.net/"
       
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.collection_name = os.getenv("MONGO_COLLECTION")
        



    def initialize_vectorstore(self):
        """
        Set up the retriever, which will retrieve the relevant snippets of text from the blog.
        """
        print("Initializing vectorstore...")
        
        
        vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            self.uri,
            self.db_name + "." + self.collection_name,
            OpenAIEmbeddings(disallowed_special=()),
            index_name="opinion_index",
        )

        return vectorstore


    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]


    def initialize_prompt(self):
        print("initializing prompt")
        contextualize_q_system_prompt = """Given a opinion, chat history and latest user question \
        which might reference context in the chat history. Keep the context as it is. DO NOT answer the question, \
        just reformulate the question if needed and otherwise return it as is.
        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        return contextualize_q_prompt, chat_prompt

    def create_chain(self):

        print("creating chain...")
        llm_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
        )
        vectorstore = self.initialize_vectorstore()

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        contextualize_q_prompt, chat_prompt = self.initialize_prompt()

        history_aware_retriever = create_history_aware_retriever(
            llm_model, retriever, contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(
            llm_model, chat_prompt
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
        print("created chain")
    
    def get_relevant_docs(self):
        vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            self.uri,
            self.db_name + "." + self.collection_name,
            OpenAIEmbeddings(disallowed_special=()),
            index_name="opinion_index",
        )
        relevant_docs = vectorstore.similarity_search(query="query", k=1)
        return relevant_docs

