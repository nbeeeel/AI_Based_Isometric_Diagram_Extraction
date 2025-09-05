from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

model = YOLO('best.pt')

input_dir = ('images')
output_dir = ('output')

os.makedirs(output_dir,exist_ok=True)

img_path = (r'D:\diagram_output\diagram_output\Test Pictures\334D92461.jpg')

output_dir = ('output/test')
img = cv2.imread(img_path)

results = model(img)

# # Define the custom metric
def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
model_constraint = tf.keras.models.load_model('anomaly.keras')
model1 = tf.keras.models.load_model('my_model_keras.keras', custom_objects={'MSE': MSE})

def calculate_reconstruction_error(original, reconstructed):
    return 1/np.mean(np.square(original - reconstructed), axis=(1, 2, 3))  # Assuming 4D images (batch_size, width, height, channels)

threshold = 0.014581064133542713 #Acquired from the model training, refer to google colab notebook

for i, result in enumerate(results[0].boxes.xyxy):
    x_min, y_min, x_max, y_max = map(int, result)
    cropped_img = img[y_min:y_max, x_min:x_max]
    cropped_img = cv2.resize(cropped_img,(224,224))
    cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_RGB2GRAY)
    cropped_img = np.expand_dims(cropped_img,axis=0)
    check = model_constraint.predict(cropped_img)
    error_recon = calculate_reconstruction_error(cropped_img,check)
    print('Reconstruction Error For Input Image :',error_recon)
    if (error_recon < threshold):
        print('Isometric Image Succesfully Detected')
        cleaned = model1.predict(cropped_img / 255.0)

        # Normalize back to [0,255] and convert to uint8
        cleaned_img = (cleaned[0] * 255).astype(np.uint8)

        # Ensure it's 2D (grayscale)
        if cleaned_img.ndim == 3 and cleaned_img.shape[-1] == 1:
            cleaned_img = cleaned_img.squeeze(-1)

        output_path ='D:/diagram_output/diagram_output/output'
        # Save properly
        save_path = os.path.join(output_path, 'cleaned1.jpg')
        cv2.imwrite(save_path, cleaned_img)
        print(f"Saved cleaned image to {save_path}")

    else:
        print('Non-Isometric Image Detected')



