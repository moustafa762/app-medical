import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, request, session, redirect, url_for, flash

app = Flask(__name__,template_folder='/home/ahisham/Downloads/keras_recovery/app_medical/templates/')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
def get_model():
    global model
    model = load_model('/home/ahisham/Downloads/keras_recovery/app_medical/models/chestxray_vgg16.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()


   


@app.route("/static/predict", methods=["POST"])
@cross_origin()
def predict():
    
    #return render_template('index.html')
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'covid19': prediction[0][0],
            'lung_opacity': prediction[0][1],
            'Normal' : prediction[0][2],
            'Viral_pnemounia' : prediction[0][3]
        }
    }
    return jsonify(response),render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
