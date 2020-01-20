import os
from threading import Thread
from flask import Flask, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from finetune import train_custom

UPLOAD_FOLDER = './mydata'
ALLOWED_EXTENSIONS = set(['.tsv'])

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/createWebApp', methods=['POST'])
def webapp():
	title = request.args.get('text')

	if not os.path.exists('./my-app-'+title):
		os.makedirs('./my-app-'+title)






@app.route('/trainAndstream', methods=['POST'])
def trainAndstream():
	file = request.files['filename']
	filename = file.filename
	filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	file.save(filepath)

	def do_work():
		train_custom(filepath)

	thread = Thread(target=do_work)
	thread.start()
	thread.join()
	return '1'

app.run(debug=True)
