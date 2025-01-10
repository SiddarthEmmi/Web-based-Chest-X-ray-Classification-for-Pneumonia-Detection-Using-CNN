from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'uploads'
TRAINED_MODEL_PATH = 'chest_xray_resnet50_model.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Define helper function to check for chest X-ray
def is_chest_xray(image_path):
    img = Image.open(image_path)
    if img.mode != 'L':  # Ensure grayscale mode
        return False
    return True

# Load or train the model
def train_model():
    img_width, img_height = 224, 224  # Adjust image size for ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Freeze layers
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Data augmentation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255, preprocessing_function=preprocess_input)
    
    # Load data (Replace with your data paths)
    train_generator = train_datagen.flow_from_directory(
        'S:\\BITM\\6TH SEM\\MINI PROJECT\\train',
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        'S:\\BITM\\6TH SEM\\MINI PROJECT\\val',
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary'
    )
    
    # Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=5)
    model.save(TRAINED_MODEL_PATH)

# Train or load the model
if not os.path.exists(TRAINED_MODEL_PATH):
    train_model()
model = load_model(TRAINED_MODEL_PATH)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file provided."
    
    file = request.files['image']
    if file.filename == '':
        return "No file selected."
    
    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    
    # Check if image is a chest X-ray
    if not is_chest_xray(image_path):
        result = "Uploaded image does not appear to be a chest X-ray."
        return render_template('result.html', result=result, image_path=file.filename, is_xray=False)
    
    # Preprocess the image
    img = load_img(image_path, target_size=(64, 64))  # Resize to 64x64
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize as in training
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    return render_template('result.html', result=result, image_path=file.filename, is_xray=True)


if __name__ == '__main__':
    app.run(debug=True)
