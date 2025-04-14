"""from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import os
import logging



app = Flask(__name__, template_folder='views')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
model_path = "C:\\sem-6\\mi2\\Detection-of-Diabetic-Retinopathy-using-CNN-main\\model_last.h5"
if not os.path.exists(model_path):
    logging.error(f'File {model_path} does not exist.')
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('start.hbs')

@app.route('/login')
def login():
    return render_template('login.hbs')

@app.route('/category')
def category():
    return render_template('category.hbs')

@app.route('/hospital')
def hospital():
    return render_template('hospital.hbs')

@app.route('/medication')
def Medical():
    return render_template('medication.hbs')

@app.route('/Profile')
def profile():
    return render_template('Profile.hbs')

@app.route('/checkup')
def check():
    return render_template('checkup.hbs')





@app.route('/signup')
def sign():
    return render_template('signup.hbs')

@app.route('/home')
def home():
    return render_template('home.hbs')

def preprocess_image(file):
    img_bytes = BytesIO(file.read())
    img = image.load_img(img_bytes, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img, img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        
        # Number of predictions to average
        num_predictions = 5
        predictions = []

        img, img_array = None, None  # Initialize variables

        for _ in range(num_predictions):
            img, img_array = preprocess_image(file)
            prediction = model.predict(img_array)
            predictions.append(prediction)
            file.seek(0)  # Reset file pointer to the beginning

        # Average the predictions
        avg_prediction = np.mean(predictions, axis=0)
        
        # Map predictions to the respective classes
        classes = ["MILD", "MODERATE", "NO DR", "PROLIFERATE DR", "SEVERE"]
        predicted_class = classes[np.argmax(avg_prediction)]

        # Pass the image and prediction result to the template
        return render_template('checkup.hbs', prediction=predicted_class, img=img)

    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
# three image displayed 
"""
#   #############################################################################################################
# from flask import Flask, render_template, request, send_file
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from io import BytesIO
# import os
# import logging
# from PIL import Image, ImageOps
# import base64

# app = Flask(__name__, template_folder='views')

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# # Load the pre-trained model
# model_path = "model_last.h5"
# if not os.path.exists(model_path):
#     logging.error(f'File {model_path} does not exist.')
# model = load_model(model_path)

# @app.route('/')
# def index():
#     return render_template('start.hbs')

# @app.route('/login')
# def login():
#     return render_template('login.hbs')

# @app.route('/category')
# def category():
#     return render_template('category.hbs')

# @app.route('/hospital')
# def hospital():
#     return render_template('hospital.hbs')

# @app.route('/medication')
# def Medical():
#     return render_template('medication.hbs')

# @app.route('/Profile')
# def profile():
#     return render_template('Profile.hbs')

# @app.route('/checkup')
# def check():
#     return render_template('checkup.hbs')

# @app.route('/signup')
# def sign():
#     return render_template('signup.hbs')

# @app.route('/home')
# def home():
#     return render_template('home.hbs')

# def preprocess_image(file):
#     img_bytes = BytesIO(file.read())
#     img = image.load_img(img_bytes, target_size=(64, 64))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize the image
#     return img, img_array

# def convert_image_to_base64(img):
#     buffered = BytesIO()
#     img.save(buffered, format="PNG")
#     img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return img_base64

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the image file from the request
#         file = request.files['file']
        
#         # Preprocess the image
#         img, img_array = preprocess_image(file)
        
#         # Create grayscale version of the image
#         img_gray = ImageOps.grayscale(img)
        
#         # Perform prediction
#         prediction = model.predict(img_array)
        
#         # Map predictions to the respective classes
#         classes = ["MILD", "MODERATE", "NO DR", "PROLIFERATE DR", "SEVERE"]
#         predicted_class = classes[np.argmax(prediction)]
        
#         # Convert images to base64
#         uploaded_image_base64 = convert_image_to_base64(img)
#         preprocessed_image_base64 = convert_image_to_base64(Image.fromarray((img_array[0] * 255).astype(np.uint8)))
#         gray_image_base64 = convert_image_to_base64(img_gray)
        
#         # Pass the images and prediction result to the template
#         return render_template('checkup.hbs', 
#                                prediction=predicted_class, 
#                                uploaded_image=uploaded_image_base64, 
#                                preprocessed_image=preprocessed_image_base64, 
#                                gray_image=gray_image_base64)

#     except Exception as e:
#         logging.error(f'Error during prediction: {str(e)}')
#         return str(e)

# if __name__ == '__main__':
#     app.run(debug=True)

#####################################################################################################################################

from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import os
import logging
from PIL import Image, ImageOps
import base64

app = Flask(__name__, template_folder='views')
app.secret_key = "your_secret_key"  # Needed for flashing messages

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
model_path = "model_last.h5"
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
else:
    logging.warning(f"Model file '{model_path}' not found. Predictions will not work.")

# Dummy user storage (Replace this with actual database logic)
users = {
    "testuser": "testpassword"  # Example user (replace with a database)
}

@app.route('/')
def index():
    return render_template('start.hbs')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        if username in users and users[username] == password:
            logging.info(f"User {username} logged in successfully.")
            return redirect(url_for('home'))  # Redirect to home on successful login
        else:
            flash("Invalid username or password!", "error")
            logging.warning(f"Failed login attempt for user: {username}")
            return redirect(url_for('login'))

    return render_template('login.hbs')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("Please fill in all fields!", "error")
            return redirect(url_for('signup'))

        if username in users:
            flash("Username already exists. Try another one.", "error")
            return redirect(url_for('signup'))

        users[username] = password  # Store user (replace with database logic)
        logging.info(f"New user signed up: {username}")

        flash("Signup Successful! Please login.", "success")
        return redirect(url_for('login'))  # Redirect to login after signup
    
    return render_template('signup.hbs')

@app.route('/category')
def category():
    return render_template('category.hbs')

@app.route('/hospital')
def hospital():
    return render_template('hospital.hbs')

@app.route('/medication')
def Medical():
    return render_template('medication.hbs')

@app.route('/Profile')
def profile():
    return render_template('Profile.hbs')

@app.route('/checkup')
def check():
    return render_template('checkup.hbs')

@app.route('/home')
def home():
    return render_template('home.hbs')

def preprocess_image(file):
    """ Preprocesses an uploaded image file for model prediction. """
    img_bytes = BytesIO(file.read())
    img = image.load_img(img_bytes, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img, img_array

def convert_image_to_base64(img):
    """ Converts an image to a Base64 string for displaying in HTML. """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash("Model not loaded. Prediction is unavailable.", "error")
        return redirect(url_for('check'))

    try:
        # Ensure a file was uploaded
        if 'file' not in request.files or request.files['file'].filename == '':
            flash("No file uploaded!", "error")
            return redirect(url_for('check'))

        file = request.files['file']
        
        # Preprocess the image
        img, img_array = preprocess_image(file)
        
        # Convert to grayscale
        img_gray = ImageOps.grayscale(img)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Define class labels
        classes = ["MILD", "MODERATE", "NO DR", "PROLIFERATE DR", "SEVERE"]
        predicted_class = classes[np.argmax(prediction)]
        
        # Convert images to base64
        uploaded_image_base64 = convert_image_to_base64(img)
        preprocessed_image_base64 = convert_image_to_base64(Image.fromarray((img_array[0] * 255).astype(np.uint8)))
        gray_image_base64 = convert_image_to_base64(img_gray)
        
        return render_template('checkup.hbs', 
                               prediction=predicted_class, 
                               uploaded_image=uploaded_image_base64, 
                               preprocessed_image=preprocessed_image_base64, 
                               gray_image=gray_image_base64)

    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        flash("An error occurred during prediction!", "error")
        return redirect(url_for('check'))

if __name__ == '__main__':
    app.run(debug=True)


"""

from flask import Flask, render_template, request, send_file
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, model_from_json
import numpy as np
from io import BytesIO
import os
import logging
from PIL import Image, ImageOps
import base64
import matplotlib.pyplot as plt
import json

app = Flask(__name__, template_folder='views')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
model_path = "C:\\sem-6\\mi2\\Detection-of-Diabetic-Retinopathy-using-CNN-main\\model_last.h5"
if not os.path.exists(model_path):
    logging.error(f'File {model_path} does not exist.')
model = load_model(model_path)

model_json = model.to_json()

with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)


model = model_from_json(model_json)


with open('model_architecture.json', 'r') as f:
    history = json.load(f)


@app.route('/')
def index():
    return render_template('start.hbs')

@app.route('/login')
def login():
    return render_template('login.hbs')

@app.route('/category')
def category():
    return render_template('category.hbs')

@app.route('/hospital')
def hospital():
    return render_template('hospital.hbs')

@app.route('/medication')
def Medical():
    return render_template('medication.hbs')

@app.route('/Profile')
def profile():
    return render_template('Profile.hbs')

@app.route('/checkup')
def check():
    return render_template('checkup.hbs')

@app.route('/signup')
def sign():
    return render_template('signup.hbs')

@app.route('/home')
def home():
    return render_template('home.hbs')

def preprocess_image(file):
    img_bytes = BytesIO(file.read())
    img = image.load_img(img_bytes, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img, img_array

def convert_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(buffer)
    buffer.seek(0)
    return buffer

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        
        # Preprocess the image
        img, img_array = preprocess_image(file)
        
        # Create grayscale version of the image
        img_gray = ImageOps.grayscale(img)
        
        # Perform prediction
        prediction = model.predict(img_array)
        
        # Map predictions to the respective classes
        classes = ["MILD", "MODERATE", "NO DR", "PROLIFERATE DR", "SEVERE"]
        predicted_class = classes[np.argmax(prediction)]
        
        # Convert images to base64
        uploaded_image_base64 = convert_image_to_base64(img)
        preprocessed_image_base64 = convert_image_to_base64(Image.fromarray((img_array[0] * 255).astype(np.uint8)))
        gray_image_base64 = convert_image_to_base64(img_gray)
        
        # Create training history plot
        history_plot_buffer = plot_training_history(history)
        history_plot_base64 = base64.b64encode(history_plot_buffer.getvalue()).decode('utf-8')
        
        # Pass the images, prediction result, and history plot to the template
        return render_template('checkup.hbs', 
                               prediction=predicted_class, 
                               uploaded_image=uploaded_image_base64, 
                               preprocessed_image=preprocessed_image_base64, 
                               gray_image=gray_image_base64,
                               history_plot=history_plot_base64)

    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
"""