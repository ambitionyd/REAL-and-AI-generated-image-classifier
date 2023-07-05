import streamlit as st
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import requests,os

def download_model():
    if not os.path.exists("model.h5"):
        response=requests.get("https://rahimcdn.blob.core.windows.net/mycdn/model.h5")
        with open("model.h5","wb") as file:
            file.write(response.content)

def load_model():
    download_model()
    model=keras.models.load_model("model.h5")
    return model

model=load_model()
def predict():
    img=keras.utils.load_img(file,target_size=(256,256))
    img=keras.utils.img_to_array(img)
    img=tf.expand_dims(img,0)
    pred=model.predict(img)
    if np.argmax(tf.nn.softmax(pred[0]))==0:
        st.write("AI Generated")
    else:
        st.write("Real")

st.title("AI Image Classifier")
with st.spinner("Loading"):
    file=st.file_uploader("Upload Image",accept_multiple_files=False,type=['jpg','png','jpeg'])
if file is not None:
    content=file.getvalue()
    st.image(content)
    with st.spinner("Predicting"):
        predict()


