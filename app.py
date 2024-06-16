import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model('cat_dog_classifier.h5')


def predict_class(img):
    if img is None:
        st.error("No image provided.")
        return None

    img = img.resize((150, 150))  # Changed: Resize directly using PIL
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    print("Prediction:", prediction)
    if prediction[0][0] > 0.5:
        return "Dog"
    return "Cat"

st.markdown(
    """
    <style>
    h2{
        text-align: center;
        font-family: 'poppins', sans-serif;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #000;
        margin-top: 2rem;
    }
    .copyright {
        text-align: center;
        font-size: 0.8rem;
        color: #000;
        margin-top: 1rem;
    }
    .footer a{
        font-family: 'poppins', sans-serif;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<title>Cat-Dog Classifier</title> <div class="main"> <h2>Cat vs Dog Classifier</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    st.image(image, caption='Uploaded Image.', use_column_width=False)

    if st.button("Predict"):
        emotion = predict_class(image)
        if emotion:
            st.success(f"Predicted Emotion: {emotion}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('''
    <div class="footer">
        Developed by Prabhat Kumar Raj
        <br>
        <a href="https://github.com/Prabhat-2101/Cat_Dog_Classifier_App_Using_Custom_CNN" target="_blank">View on GitHub</a>
    </div>
    <div class="copyright">
        &copy; 2024 All rights reserved @moodscannerai
    </div>
    ''', unsafe_allow_html=True)