# import part
from transformers import pipeline
from PIL import Image
import streamlit as st

# function part
def ageClassifier():
    # Load the age classification pipeline
    # The code below should be placed in the main part of the program
    age_classifier = pipeline("image-classification",
                              model="dima806/fairface_age_image_detection")
    
    image_name = "middleagedMan.jpg"
    image_name = Image.open(image_name).convert("RGB")
    
    # Classify age
    age_predictions = age_classifier(image_name)

    return age_predictions

def main():
    
    # Streamlit UI
    st.header("Title: Age Classification using ViT")

    age_predictions = ageClassifier():
    
    st.write(age_predictions)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
    
    # Display results
    st.write("Predicted Age Range:")
    st.write(f"Age range: {age_predictions[0]['label']}")

# main part
if __name__ == "__main__":
    main()




