import os
import cv2
import keras
import json
import numpy as np
from flask import Flask,request,redirect
from keras.models import load_model
from werkzeug import secure_filename

app = Flask(__name__)

UPLOAD = 'static/upload'
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
    filename = os.path.join(UPLOAD , filename)
    f.save(os.path.join(app.root_path , filename))
    classname = predict_image(model , filename)
    img = predict_ndarray(filename)
    op = model.predict(np.array([img]))
    classpre = 0.0
    # print(op)
    for i in op:
        # 取最大的概率
        # print(list(i)[0].tolist())
        classpre = max(list(i)).tolist()
        # print(classpre)
    filelist.append({'filename':f.filename,'classname':classname , 'predict':round(classpre,2)})
    # print(type(filelist))
    return json.dumps(filelist)

@app.before_first_request
def create():
    path = os.path.join(app.root_path,UPLOAD)
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    app.run('0.0.0.0')