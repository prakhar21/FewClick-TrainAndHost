from time import sleep
from flask import Flask, render_template
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trainAndstream')
def trainAndstream():
    def generate():
        with open('./model/training.log') as f:
            while True:
                yield f.read()
                sleep(1.5)

    return app.response_class(generate(), mimetype='text/plain')

app.run(debug=True)
