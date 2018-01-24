import os
import cv2
from flask import Flask, request, redirect, url_for,send_file, flash
from werkzeug.utils import secure_filename
import matplotlib.image as mpimg
from werkzeug.datastructures import ImmutableMultiDict


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'wuey24y723urheqwkukqje2131j3j1k3hk1jgrer123123wqeq'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process', methods=['POST'])
def upload_file():
    
	# check if the post request has the file part
	if 'image' not in request.files:
	    flash('No file part')
	    return redirect(request.url)
	
	file = request.files['image']
	# if user does not select file, browser also
	# submit a empty part without filename

	if file.filename == '':
	    flash('No selected file')
	    return redirect(request.url)

	else: #and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		## Read the saved file
		metadata = os.path.join(app.config['UPLOAD_FOLDER'],"config.txt")

		os.system("python ImageStitching/main.py {0}".format(metadata))
	    
	return send_file("output/app_output.jpg",mimetype="image/jpeg") 
