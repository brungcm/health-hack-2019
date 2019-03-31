# coding=utf-8

import json
import logging
import os

from flask import Flask, jsonify

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

app = Flask(__name__)


@app.route('/')
def default():
    return 'People-Tracker-App OK'


@app.route('/status')
def get_status():
    filepath = os.environ['STATUS_FILE']
    with open(filepath, 'r') as status_file:
        status_data = json.load(status_file)
    return jsonify(status_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
