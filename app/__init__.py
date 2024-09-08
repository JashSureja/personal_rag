import os

from flask import Flask
from .routes import main

def create_app(debug=False):
    app = Flask(__name__)
    from app.langchain import LangChain
    app.secret_key = os.getenv('SECRET_KEY')
    app.register_blueprint(main)
    langchain = LangChain()

    if debug:
        app.config['DEBUG'] = True


    app.config["LANGCHAIN"] = langchain
    return app