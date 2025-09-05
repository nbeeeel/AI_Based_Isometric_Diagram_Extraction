## Isometric Diagram Extraction and Extraneous Details Removal 🖼️
This project implements a sophisticated image processing pipeline combining YOLOv8 for object detection with TensorFlow/Keras models for anomaly detection and image reconstruction. It identifies isometric images, evaluates anomalies, and generates cleaned outputs, optimized for grayscale processing. A preprocessing notebook is included for dataset preparation. 🚀 
### Installation

You can install the latest release from PyPI:
pip install ai-isometric-extractor==0.1.2

### Table of Contents 📑

- Overview 🌐
- Features ✨
- Project Structure 📂
- Requirements 🛠️
- Installation ⚙️
- Usage 📋
- Preprocessing Dataset 📊
- Running the Main Script ▶️

## Model Details 🤖
- YOLO Model 🔍
- Anomaly Detection Model ⚠️
- Image Reconstruction Model 🖌️

- Dataset 📸
- Threshold Configuration 📏
- Output 💾
- Contributing 🤝
- License 📜


### Overview 🌐
This project leverages YOLOv8 to detect regions of interest in images and employs two custom-trained TensorFlow/Keras models to:

- Classify regions as isometric or non-isometric using a reconstruction error metric.
- Reconstruct valid isometric images into cleaned grayscale outputs.

The pipeline processes images from an input directory, applies anomaly detection with a custom Mean Squared Error (MSE) metric, and saves cleaned images if they meet isometric criteria. The Diagram_Constraints.ipynb notebook prepares the dataset by resizing and normalizing images. 🧠

### Features ✨

- 🔍 Object Detection: YOLOv8 (best.pt) identifies regions of interest.
- ⚠️ Anomaly Detection: Classifies images using a custom MSE metric.
- 🖌️ Image Reconstruction: Generates cleaned grayscale images for isometric inputs.
- 📊 Preprocessing Pipeline: Jupyter notebook for dataset preparation.
- 📏 Customizable Threshold: Adjustable reconstruction error threshold.
- 🖼️ Grayscale Processing: Optimized for efficient grayscale image handling.


### Project Structure 📂
isometric-image-detection/ <br>
├── main.py                   # 🖥️ Main script for detection and reconstruction <br>
├── Diagram_Constraints.ipynb # 📓 Notebook for dataset preprocessing <br>
├── images/                   # 📥 Input directory for test images <br>
├── output/                   # 📤 Output directory for cleaned images <br>
├── best.pt                   # 🤖 Pre-trained YOLOv8 model <br>
├── anomaly.keras             # ⚠️ Pre-trained anomaly detection model <br>
├── my_model_keras.keras      # 🖌️ Pre-trained image reconstruction model <br>
├── README.md                 # 📜 Project documentation <br>

### Requirements 🛠️
To run this project, install the following dependencies:

- Python 3.8+ 🐍
- ultralytics
- tensorflow
- numpy
- opencv-python
- matplotlib
- jupyter (for the preprocessing notebook)

### Install them using:
pip install ultralytics tensorflow numpy opencv-python matplotlib jupyter

### Installation ⚙️
- Clone the Repository:
- git clone https://github.com/your-username/isometric-image-detection.git
- cd isometric-image-detection

### Install Dependencies:
pip install -r requirements.txt

### Download Pre-trained Models:
Place best.pt, anomaly.keras, and my_model_keras.keras in the project root. 📥
These models are assumed pre-trained from your training environment (e.g., Google Colab).

### Set Up Directories:
- Create an images/ directory for input images.
- Create an output/ directory for processed images.

### Usage 📋
- Preprocessing Dataset 📊
- The Diagram_Constraints.ipynb notebook prepares the dataset for training. Steps:

#### Open the Notebook:
jupyter notebook Diagram_Constraints.ipynb

#### Configure Paths:
- Update paths for Isometric_Images and Non_Isometric to your dataset directories:Isometric_Images = '/path/to/ground_truth/Or'
- Non_Isometric = '/path/to/diag'

#### Run the Notebook:
- Loads images, resizes to 224x224, converts to grayscale, and normalizes to [0,1].
- Visualizes samples with matplotlib for verification. 📈

#### Running the Main Script ▶️
The main.py script processes images for detection and reconstruction:

##### Update Image Path:
Edit img_path in main.py:img_path = 'path/to/your/image.jpg'

##### Run the Script:
python main.py

### Process:
- YOLOv8 detects regions. 🔍
- Crops, resizes (224x224), and converts to grayscale.
- anomaly.keras computes reconstruction error. ⚠️
- If error < threshold (0.014581064133542713), classifies as isometric and reconstructs using my_model_keras.keras. 🖌️
- Saves cleaned images to output/ (e.g., cleaned1.jpg). 💾

### Model Details 🤖
- YOLO Model 🔍
- File: best.pt
- Purpose: Detects regions of interest using YOLOv8.
- Output: Bounding box coordinates (xyxy).

### Anomaly Detection Model ⚠️
- File: anomaly.keras
- Purpose: Classifies images as isometric via reconstruction error.
- Metric: Custom MSE:def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

### Image Reconstruction Model 🖌️
- File: my_model_keras.keras
- Purpose: Reconstructs valid isometric images.
- Input: Normalized grayscale images (224x224).
- Output: Cleaned grayscale images saved as uint8.

### Dataset 📸
- Isometric Images: Stored in Isometric_Images (e.g., /content/drive/MyDrive/ground_truth/Or).
- Non-Isometric Images: Stored in Non_Isometric (e.g., /content/drive/MyDrive/diag).

#### Preprocessing:
- Resized to 224x224, converted to grayscale.
- Normalized to [0,1].
- Expanded with channel dimension for model compatibility.

### Threshold Configuration 📏
- Value: 0.014581064133542713
- Source: Derived from training (see Google Colab notebook).
- Purpose: Classifies images as isometric if error_recon < threshold.
- Adjustment: Modify threshold in main.py based on validation performance.

### Output 💾
- Directory: output/ (configurable in main.py).
- Format: Cleaned images saved as .jpg (e.g., cleaned1.jpg).
- Content: Reconstructed grayscale images for isometric inputs.

#### Contributing 🤝
Contributions are welcome! To contribute:
- Fork the repository. 🍴
- Create a branch: git checkout -b feature/your-feature.
- Commit changes: git commit -m 'Add your feature'.
- Push: git push origin feature/your-feature.
- Open a Pull Request. 📬

Ensure code follows the project's style and includes documentation.

License 📜
This project is licensed under the MIT License. See the LICENSE file for details.
