#!flask/bin/python
"""Wrap and expose API interfaces for sentiment analysis."""

from flask import Flask, jsonify
from RandomForestsSenAnalysis import invoke_rf_test, invoke_rf_train

application = Flask(__name__)

@application.route('/')

def index():
    """Random forest test api."""
    accuracy = invoke_rf_test()
    response = jsonify({"Accuracy" : accuracy})

    return response

@application.route('/rf_train')

def rf_train():
    """Random forest training api."""
    invoke_rf_train(5, 10)
    response = jsonify({"Result" : "Training finished"})

    return response

if __name__ == '__main__':
    application.run(debug=True)

