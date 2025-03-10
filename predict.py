import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # i added this after receiving some warnings 
import argparse
import tensorflow as tf
import numpy as np
import json 
from tensorflow import keras
from PIL import Image
import tensorflow_hub as hub

# process func
def process_image(image_path):
    # match input format to model format
    image = Image.open(image_path).resize((224, 224)) 
    image = np.array(image) / 255.0  # normalize
    return np.expand_dims(image, axis=0)  # add batch

# predict func
def predict(image_path, model_path, top_k=1, category_names=None):
    # it will: load the model i saved -> process the image chosen in command line-> make predictions
    # first: load model
    model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # second: process input image
    processed_image = process_image(image_path)
    
    # third: make predictions
    predictions = model.predict(processed_image)[0]
    
    # top K predictions
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]
    
    # load category names 
    if category_names:
        with open(category_names, 'r') as f:
            label_map = json.load(f)
        top_k_labels = [label_map.get(str(i + 1), f"Class {i + 1}") for i in top_k_indices]
    else:
        top_k_labels = [f"Class {i + 1}" for i in top_k_indices]
    
    return list(zip(top_k_labels, top_k_probs))

# main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict flower species from an image using the trained model")
    parser.add_argument("image_path", type=str, help="path to the image")
    parser.add_argument("model_path", type=str, help="path to the trained model")
    parser.add_argument("--top_k", type=int, default=1, help="return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="path to JSON file mapping class indices to names")
    
    args = parser.parse_args()
    
    # run prediction
    predictions = predict(args.image_path, args.model_path, args.top_k, args.category_names)
     
    # results
    for i, (label, prob) in enumerate(predictions, 1):
        print(f"{i}: {label} ({prob * 100:.2f}%)")
