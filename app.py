import os
from flask import Flask, redirect

app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/mtcnn')

if __name__ == '__main__':
    app.run(debug=True) 