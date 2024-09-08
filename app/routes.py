from flask import Blueprint, render_template, request, session, current_app

import os


main = Blueprint("main", __name__)

conversation_history = []

@main.route("/")
def home():
    
    return render_template("index.html")


@main.route("/login", methods=["POST"])
def login():
    password = request.form["password"]

    if password == os.getenv("SESSION_PASSWORD"):
        session["logged_in"] = True
        global conversation_history
        langchain = current_app.config["LANGCHAIN"]
        conversation_history = []
        langchain.create_chain()
        return render_template("chat.html", history=conversation_history)


@main.route("/chat", methods=["POST", "GET"])
def process_form():
    if session.get("session_id") is None:
        session_id = os.urandom(2)
    else:
        session_id = session.get("session_id")
    langchain = current_app.config["LANGCHAIN"]
    user_question = session.get("question")
    conversation_history.append(("User", user_question))

    response = langchain.chain_with_message_history.invoke(
        {"input": user_question},
        {"configurable": {"session_id": session_id}},
    )
    conversation_history.append(("Model", response["answer"]))
    print(response)
    return render_template("chat.html", history=conversation_history)


@main.route("/new")
def reset():
    global conversation_history
    conversation_history = []
    session_id = os.urandom(2)
    session["session_id"] = session_id
    
    return render_template("chat.html", history=conversation_history)


@main.route("/error")
def error():
    return render_template("error.html")


@main.route("/load", methods=["POST"])
def show_chat():
    session["question"] = request.form["question"]
    return render_template("loading.html", history=conversation_history)

