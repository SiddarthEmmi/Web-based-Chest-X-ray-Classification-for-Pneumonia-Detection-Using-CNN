# Web-based-Chest-X-ray-Classification-for-Pneumonia-Detection-Using-CNN
This project is a Flask-based web application that uses a Convolutional Neural Network (CNN) for detecting pneumonia from chest X-ray images. The system allows users to upload chest X-ray images, processes the images through a pre-trained ResNet50 model, and predicts whether the image indicates "Pneumonia" or "Normal". The project combines deep learning with web development to provide a user-friendly interface for medical imaging analysis.

# Key Features
* Deep Learning: Uses ResNet50 pre-trained on ImageNet, fine-tuned for binary classification (Pneumonia vs. Normal).
* Web Interface: Built with Flask, providing an easy-to-use platform for uploading and analyzing chest X-rays.
* Image Validation: Includes a helper function to ensure the uploaded image is a valid chest X-ray.
* Dynamic Visualization: Displays the uploaded image along with the prediction result on the result page.

# Technical Details
* Backend: Python, Flask
* Deep Learning Framework: TensorFlow/Keras
* Preprocessing: Image resizing, normalization, and data augmentation (during training).
* Model Architecture: ResNet50 with added dense layers for binary classification.
* Data Handling: Uses TensorFlow's ImageDataGenerator for training and validation.

# Dataset
The dataset consists of 4,538 chest X-ray images, divided into two categories:

* Normal: Healthy lungs
* Pneumonia: Lungs affected by pneumonia
Each category contains images organized into respective folders. The dataset is preprocessed to convert the images to a suitable size and normalized to enhance model performance.

* dataset: You can download the dataset by this link - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Project Setup
* Clone the repository:
        git clone <repository-link>
        cd <repository-name>
* Install the dependencies:
        pip install -r requirements.txt
* Place your trained model file (chest_xray_resnet50_model.h5) in the project directory or modify the script to train a new model.
* Run the application:
        python app.py
* Access the app at http://127.0.0.1:5000 in your browser.

# Future Enhancements
* Deploying the application on a cloud platform (e.g., AWS, Heroku, or Azure).
* Adding more robust validation for chest X-ray images.
* Extending functionality to multi-class classification for detecting other lung conditions.

# Project Associates:
* Siddartha S Emmi
* Shaik Mohaddis
* Shashank N M
* Shreeraja H M
