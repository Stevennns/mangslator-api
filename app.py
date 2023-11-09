# -*- coding: utf-8 -*-
from flask import Flask
import os

# UPLOAD_FOLDER = "/home/grupo3/mangslator-uploads/"
UPLOAD_FOLDER = "../mangslator-results/"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


DEBUG = True
