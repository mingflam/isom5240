from transformers import pipeline
from PIL import Image
import streamlit as st
from model import predict_age_gender

result = predict_age_gender("your_image.jpg")
st.write(f"Age: {result['age']}, Gender: {result['gender']}")
