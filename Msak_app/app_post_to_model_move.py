import os
import cv2
import keras
import json
import numpy as np
from flask import Flask,request,redirect
from keras.models import load_model
from werkzeug import secure_filename
import shutil

app = Flask(__name__)

UPLOAD = 'static/upload'
UPLOADHOW = 'static/upload/WithMask'
UPLOADNO = 'static/upload/WithoutMask'
FILWMOVE = 'static/movefile'
ALLOERD = {'jpg','jpeg','png','gif'}

@app.route('/')
def index():
    return app.send_static_file('upload.html')

model_path = 'static/model_119-07-0.10-0.97.hdf5'
model = load_model(model_path , compile=False)

def allowerd_file(filename):
    return '.' in filename and filename.rsplit('.' , 1)[1].lower() in ALLOERD

def classId2className(classId):
    return ['WithMask','WithoutMask'][classId]

def predict_ndarray(img_path):
    img = cv2.resize(cv2.imread(img_path),(64,64),interpolation=cv2.INTER_AREA)
    img = img.astype('float32')
    return np.array(img[:,:,:3]/255.)

def predict_image(model , imgPath):
    img = predict_ndarray(imgPath)
    inputs = np.argmax(model.predict(np.array([img])))
    predict_classname = classId2className(inputs)
    return predict_classname

@app.route('/upload',methods=['POST'])
def post_upload():
    filelist = []
    if 'file' not in request.files:
        return redirect('/')
    f = request.files['file']
    if not f or not allowerd_file(f.filename):
        return 'o no'
    filename = secure_filename(f.filename)
    filemove = os.path.join(FILWMOVE , filename)
    f.save(os.path.join(app.root_path , filemove))
    classname = predict_image(model , filemove)
    img = predict_ndarray(filemove)
    op = model.predict(np.array([img]))
    classpre = 0.0
    for i in op:
        classpre = max(list(i)).tolist()
    for file in os.listdir(UPLOAD):
        if file == classname:
            file_d = os.path.join(UPLOAD , file)
            filename = os.path.join(file_d ,filename)
            shutil.move(filemove , filename)
            break
    filelist.append({'filename':f.filename,'classname':classname , 'predict':round(classpre,2)})
    return json.dumps(filelist)

@app.before_first_request
def create():
    path = os.path.join(app.root_path,UPLOAD)
    path1 = os.path.join(app.root_path,UPLOADHOW)
    path2 = os.path.join(app.root_path,UPLOADNO)
    path3 = os.path.join(app.root_path,FILWMOVE)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path1):
        os.mkdir(path1)
    if not os.path.exists(path2):
        os.mkdir(path2)
    if not os.path.exists(path3):
        os.mkdir(path3)


if __name__ == "__main__":
    app.run('0.0.0.0')