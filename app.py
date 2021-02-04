#Import necessary libraries
from flask import Flask, render_template, request
 
import numpy as np
import os
import tensorflow as tf
 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


json_file = open('/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/model/model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('@@ Model loaded')





 
def pred_human_horse(horse_or_human):
    test_image = load_img(horse_or_human, target_size = (150, 150)) # load image 
    print("@@ Got Image for prediction")
   
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
    result =loaded_model.predict(test_image).round(3) # predict class horse or human
    print('@@ Raw result = ', result)
   
    pred = np.argmax(result) # get the index of max value
    if pred == 0:
        return "Horse" # if index 0 
    else:
        return "Human" # if index 1
 
#------------>>pred_human_horse<<--end
     








 
# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')
     
   
@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred = pred_human_horse(horse_or_human=file_path)
               
        return render_template('predict.html', pred_output = pred, user_image = file_path)
     

# #Fo AWS cloud
if __name__ == "__main__":
    app.run(host='0.0.0.0', post='8080',threaded=False) 
