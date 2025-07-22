import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import joblib

model = joblib.load('svc.pt')

def image_process(img):
    img_pil = Image.open(img).convert('RGB')
    # Resize image to match training data dimensions model expects 172800 features (160x360x3)
    img_resized = img_pil.resize((160, 360))  # Width x Height, 160*360*3=172800 
    img_arr = np.array(img_resized)
    img_flat = img_arr.flatten()
    df_t = pd.DataFrame(img_flat).T
    p = model.predict(df_t)
    return p

st.title('Shoes Classification')
file = st.file_uploader('Upload your File')
try:
    if file is not None:
        i=Image.open(file)
        st.write("Original Image:")
        st.image(i, caption=f"Original size: {i.size}")
        
        pr = image_process(file)
        st.write(f"The predicted brand is: {pr[0]}")
    else:
        st.write("Please upload an image file")
except Exception as e:
    st.error(f"Error processing image: {str(e)}")
    # Debug information
    if file is not None:
        try:
            debug_img = Image.open(file)
            debug_arr = np.array(debug_img)
            st.write(f"Debug - Image shape: {debug_arr.shape}")
            st.write(f"Debug - Flattened size: {debug_arr.flatten().shape[0]}")
            st.write("Model expects: 172800 features")
        except Exception:
            pass