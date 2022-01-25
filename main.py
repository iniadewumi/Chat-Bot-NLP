import json,pathlib
from bot_main import ChatBot
HOME = pathlib.Path().resolve()
MODELS_DIR = HOME/"MODELS"
INTENTS_DIR = HOME/"INTENTS"





# $env:FLASK_APP = "main"
# $env:FLASK_ENV = "development"
# flask run

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import json

app = Flask(__name__)



@app.route("/")
def home():
    return render_template('home.html')

chat_bot = ChatBot()


@app.route("/chat")
def chat():
    bot_response = ",".join(["0","1"])
    return render_template('index.html', data=bot_response)

@app.route("/bot/<message>")
def bot(message):
    return chat_bot.start(message)
app.run(debug=True)


