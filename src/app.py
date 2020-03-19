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


from notebook_utils.synthesize import *


try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)



@app.route("/process", methods=["POST", "GET"])
def process():

    try:
        text = request.json["text"]
        
        wav = synthesize(text, tts_model, voc_model, alpha=1.0)
        ipd.Audio(wav, rate=hp.sample_rate)

        callback = send_file(zip_output_path, mimetype='audio/wav')

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


if __name__ == '__main__':
    global camembert

    init_hparams('notebook_utils/pretrained_hparams.py')
    tts_model = get_forward_model('pretrained/forward_100K.pyt')
    voc_model = get_wavernn_model('pretrained/wave_800K.pyt')

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)

