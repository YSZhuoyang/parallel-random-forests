#!flask/bin/python
"""Wrap and expose API interfaces for sentiment analysis."""

from flask import Flask, request, jsonify
from RandomForestsSenAnalysis import (
    invoke_rf_test,
    invoke_rf_train,
    invoke_rf_analyze)

application = Flask(__name__)

@application.route('/')
def index():
    """Random forest test api."""
    return jsonify({"Response" : "This is a sentiment analysis api."})

@application.route('/rf_train')
def rf_train():
    """Random forest training api."""
    invoke_rf_train(100, 10)
    accuracy = invoke_rf_test()
    return jsonify({"Accuracy" : accuracy})

@application.route('/rf_test')
def rf_test():
    """Random forest training api."""
    accuracy = invoke_rf_test()
    return jsonify({"Accuracy" : accuracy})

@application.route('/rf_analyze')
def rf_analyze():
    """Random forest training api."""
    sentence = request.args.get('sentence')
    label = invoke_rf_analyze(sentence)
    response = jsonify({"Result" : label})
    return response

if __name__ == '__main__':
    application.run(debug=True)

