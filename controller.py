#!flask/bin/python
"""Wrap and expose API interfaces of learning algorithms."""

from flask import Flask
from RandomForestsSenAnalysis import invoke_rf

APP = Flask(__name__)

@APP.route('/rf')

def rf():
    invoke_rf(1)
    return 'rf finished'

if __name__ == '__main__':
    APP.run(debug=True)

