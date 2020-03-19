import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json


from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file
import traceback


from uuid import uuid4


from notebook_utils.synthesize import *


try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)



def generate_random_filename(upload_directory, extension):
    filename = str(uuid4())
    filename = os.path.join(upload_directory, filename + "." + extension)
    return filename


def clean_me(filename):
    if os.path.exists(filename):
        os.remove(filename)



def create_directory(path):
    os.system("mkdir -p %s" % os.path.dirname(path))



@app.route("/process", methods=["POST", "GET"])
def process():
    output_path = generate_random_filename(output_directory, "wav")

    try:
        text = request.json["text"]
        
        wav = synthesize(text, tts_model, voc_model, alpha=1.0, generate_random_filename)
        

        callback = send_file(generate_random_filename, mimetype='audio/wav')

        return callback, 200

    except:
        traceback.print_exc()
        clean_me(generate_random_filename)
        return {'message': 'input error'}, 400


if __name__ == '__main__':
    global output_directory
    global voc_model
    global tts_model

    output_directory = '/src/output/'
    create_directory(output_directory)

    init_hparams('notebook_utils/pretrained_hparams.py')
    tts_model = get_forward_model('pretrained/forward_100K.pyt')
    voc_model = get_wavernn_model('pretrained/wave_800K.pyt')

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)

