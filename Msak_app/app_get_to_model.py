import os
import cv2
import numpy as np
import keras
import json
from flask import Flask
from keras.models import load_model
import random as nd

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('image.html')


model_path = 'static/model_119-07-0.10-0.97.hdf5'
model = load_model(model_path,compile=False)

def clssId2className(classId):
    return ['WithMask','WithoutMask'][classId]

def predict_ndarray(img_path):
    img = cv2.resize(cv2.imread(img_path) , (64,64),interpolation=cv2.INTER_AREA)
    img = img.astype('float32')
    return np.array(img[:,:,:3]/255.)

def predict_image(model , imagePath):
    img = predict_ndarray(imagePath)
    inputs = np.argmax(model.predict(np.array([img])))
    predict_className = clssId2className(inputs)
    return predict_className


@app.route('/download/',methods=['GET'])
def get_img():
    path = 'static/Validation/'
    files = os.listdir(path)
    filelist = []
    for file in files:
        file_p = os.path.join(path , file)
        if os.path.isdir(file_p):
            my_files = os.listdir(file_p)
            randoms = nd.sample(my_files , 5)
            for file1 in randoms:
                file_d = os.path.join(file,file1)
                classname = predict_image(model , path + file_d)
                filelist.append({'filename':file1 , 'path':path + file_d , 'classname':classname})
    return json.dumps(filelist)


if __name__ == '__main__':
    app.run()
