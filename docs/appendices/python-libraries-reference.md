---
sidebar_position: 3
---

# Appendix C: Python Libraries Reference

## Table of Contents
1. [ROS 2 Python Libraries](#ros-2-python-libraries)
2. [Computer Vision Libraries](#computer-vision-libraries)
3. [Machine Learning Libraries](#machine-learning-libraries)
4. [Robotics Libraries](#robotics-libraries)
5. [Audio Processing Libraries](#audio-processing-libraries)
6. [Data Processing Libraries](#data-processing-libraries)
7. [Visualization Libraries](#visualization-libraries)
8. [Networking and Communication](#networking-and-communication)
9. [System and Hardware Libraries](#system-and-hardware-libraries)
10. [Utility Libraries](#utility-libraries)
11. [Installation and Setup](#installation-and-setup)
12. [Best Practices](#best-practices)

## ROS 2 Python Libraries

### Core ROS 2 Libraries

```python
# Essential imports for ROS 2 Python nodes
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32, Bool
from geometry_msgs.msg import Twist, Pose, Point, Vector3
from sensor_msgs.msg import Image, LaserScan, PointCloud2, CameraInfo
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from builtin_interfaces.msg import Time
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Example: Basic ROS 2 Node
class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')

        # Create publisher
        self.publisher = self.create_publisher(String, 'topic_name', 10)

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'input_topic',
            self.listener_callback,
            10
        )

        # Create timer
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.i = 0

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced ROS 2 Features

```python
# Services and Actions
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.duration import Duration

# Example: Parameter declaration and usage
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with descriptions
        self.declare_parameter(
            'robot_name',
            'default_robot',
            ParameterDescriptor(description='Name of the robot')
        )

        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('operating_mode', 'autonomous')

    def get_config(self):
        name = self.get_parameter('robot_name').value
        speed = self.get_parameter('max_speed').value
        mode = self.get_parameter('operating_mode').value
        return {'name': name, 'speed': speed, 'mode': mode}

# Example: Service server
from std_srvs.srv import SetBool, Trigger

class ServiceNode(Node):
    def __init__(self):
        super().__init__('service_node')

        self.srv = self.create_service(
            SetBool,
            'toggle_service',
            self.toggle_callback
        )

    def toggle_callback(self, request, response):
        self.get_logger().info(f'Request: {request.data}')
        response.success = True
        response.message = f'Toggled to {request.data}'
        return response
```

## Computer Vision Libraries

### OpenCV

```python
import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt

# Basic image processing
def basic_cv_operations(image_path):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(img, (640, 480))

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (15, 15), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    return {
        'original': img,
        'gray': gray,
        'resized': resized,
        'blurred': blurred,
        'edges': edges,
        'contours': contour_img
    }

# Feature detection
def feature_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)

    # Draw keypoints
    kp_img = cv2.drawKeypoints(img, kp, None)

    # SIFT detector (if available)
    try:
        sift = cv2.SIFT_create()
        kp_sift, des_sift = sift.detectAndCompute(gray, None)
        kp_sift_img = cv2.drawKeypoints(img, kp_sift, None)
    except:
        kp_sift_img = None

    return {
        'orb_keypoints': kp_img,
        'sift_keypoints': kp_sift_img,
        'keypoints': kp,
        'descriptors': des
    }

# Object detection with Haar Cascades
def face_detection(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return img, faces

# ArUco marker detection
def aruco_detection(img):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    if ids is not None:
        img = aruco.drawDetectedMarkers(img, corners)

    return img, corners, ids
```

### Pillow (PIL)

```python
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def pil_image_processing(image_path):
    # Open image
    img = Image.open(image_path)

    # Basic operations
    width, height = img.size
    mode = img.mode

    # Resize
    resized = img.resize((800, 600))

    # Crop
    cropped = img.crop((100, 100, 400, 400))

    # Rotate
    rotated = img.rotate(45, expand=True)

    # Convert to different mode
    grayscale = img.convert('L')
    rgba = img.convert('RGBA')

    # Apply filters
    blurred = img.filter(ImageFilter.BLUR)
    sharpened = img.filter(ImageFilter.SHARPEN)
    edge_enhanced = img.filter(ImageFilter.EDGE_ENHANCE)

    # Draw on image
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], outline='red', width=3)
    draw.text((10, 10), "Hello PIL!", fill='blue')

    return {
        'resized': resized,
        'cropped': cropped,
        'rotated': rotated,
        'grayscale': grayscale,
        'filtered': {
            'blur': blurred,
            'sharpen': sharpened,
            'edge': edge_enhanced
        },
        'annotated': img
    }

# Convert between PIL and OpenCV
def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format"""
    return np.array(pil_image)[:, :, ::-1]  # RGB to BGR

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
```

## Machine Learning Libraries

### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Basic neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop example
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Data loading
def create_dataloader(X, y, batch_size=32, shuffle=True):
    dataset = CustomDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Model saving and loading
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model_class, filepath, input_size, hidden_size, output_size):
    model = model_class(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filepath))
    return model
```

### Transformers (Hugging Face)

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import CLIPProcessor, CLIPModel
import torch

# Text classification pipeline
def text_classifier():
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this product!")
    return result

# Named Entity Recognition
def ner_pipeline():
    ner = pipeline("ner", grouped_entities=True)
    result = ner("John Doe works at Google in New York.")
    return result

# Question Answering
def qa_pipeline():
    qa = pipeline("question-answering")
    context = "The Amazon rainforest is a moist broadleaf tropical rainforest."
    question = "What type of forest is the Amazon?"
    result = qa(question=question, context=context)
    return result

# CLIP for vision-language tasks
def clip_example(image_path, text_prompts):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return probs.detach().cpu().numpy()

# Fine-tuning example
def fine_tune_model():
    from transformers import Trainer, TrainingArguments

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # binary classification
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset
    # )
    #
    # trainer.train()
```

### Scikit-Learn

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Data preprocessing
def preprocess_data(X, y=None, scale=True, encode_labels=True):
    scaler = StandardScaler() if scale else None
    label_encoder = LabelEncoder() if encode_labels and y is not None else None

    X_processed = X.copy()

    if scale:
        X_processed = scaler.fit_transform(X)

    y_processed = y
    if encode_labels and y is not None:
        y_processed = label_encoder.fit_transform(y)

    return X_processed, y_processed, scaler, label_encoder

# Classification example
def classification_example(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42)

    rf_model.fit(X_train_scaled, y_train)
    lr_model.fit(X_train_scaled, y_train)

    # Predictions
    rf_pred = rf_model.predict(X_test_scaled)
    lr_pred = lr_model.predict(X_test_scaled)

    # Evaluation
    rf_accuracy = accuracy_score(y_test, rf_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    return {
        'rf_accuracy': rf_accuracy,
        'lr_accuracy': lr_accuracy,
        'rf_report': classification_report(y_test, rf_pred),
        'rf_cm': confusion_matrix(y_test, rf_pred),
        'models': {'rf': rf_model, 'lr': lr_model},
        'scaler': scaler
    }

# Clustering example
def clustering_example(X, n_clusters=3):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return {
        'cluster_labels': cluster_labels,
        'kmeans_model': kmeans,
        'pca_model': pca,
        'pca_transformed': X_pca,
        'inertia': kmeans.inertia_
    }

# Feature selection
def feature_importance_selection(X, y, n_features=10):
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)

    return {
        'selected_features': X_selected,
        'selected_indices': selected_indices,
        'selector': selector
    }
```

## Robotics Libraries

### PyRobot

```python
# Note: PyRobot needs to be installed separately
# pip install pyrobot

try:
    from pyrobot import Robot
    import numpy as np

    def pyrobot_example():
        # Initialize robot (example with LoCoBot)
        robot = Robot('locobot')

        # Get robot state
        pose = robot.base.get_pose()
        joint_positions = robot.arm.get_joint_positions()

        # Move the robot
        robot.base.go_to_relative([0.5, 0, 0])  # Move forward 0.5m

        # Move the arm
        robot.arm.set_joint_positions([0, 0, 0, 0, 0])  # Home position

        # Get camera images
        rgb, depth = robot.camera.get_rgb_depth()

        # Close gripper
        robot.gripper.close()

        return {
            'pose': pose,
            'joints': joint_positions,
            'rgb': rgb,
            'depth': depth
        }
except ImportError:
    print("PyRobot not available")

# Custom robot interface
class SimpleRobotInterface:
    def __init__(self):
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0, 1]  # quaternion
        self.joint_angles = [0] * 6  # 6 DOF arm

    def move_to_position(self, x, y, z):
        """Move robot to absolute position"""
        self.position = [x, y, z]
        print(f"Moved to position: {self.position}")

    def move_by_offset(self, dx, dy, dz):
        """Move robot by offset"""
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz
        print(f"Moved by offset to: {self.position}")

    def get_position(self):
        """Get current position"""
        return self.position.copy()

    def set_joint_angles(self, angles):
        """Set joint angles for manipulator"""
        if len(angles) == len(self.joint_angles):
            self.joint_angles = angles.copy()
            print(f"Set joint angles: {self.joint_angles}")
        else:
            raise ValueError(f"Expected {len(self.joint_angles)} angles, got {len(angles)}")

    def get_joint_angles(self):
        """Get current joint angles"""
        return self.joint_angles.copy()
```

### Modern Robotics Library

```python
# Modern Robotics Library functions
import numpy as np

def modern_robotics_examples():
    """Examples using modern robotics concepts"""

    # 1. Transformation matrices
    def vec_to_so3(omega):
        """Convert 3-vector to so(3) matrix"""
        return np.array([[0, -omega[2], omega[1]],
                         [omega[2], 0, -omega[0]],
                         [-omega[1], omega[0], 0]])

    def matrix_exp3(so3_mat):
        """Matrix exponential for so(3)"""
        omgtheta = so3_mat
        if np.allclose(omgtheta, 0):
            return np.eye(3)
        else:
            theta = np.linalg.norm([omgtheta[2, 1], omgtheta[0, 2], omgtheta[1, 0]])
            omgmat = omgtheta / theta
            return np.eye(3) + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

    # 2. Screw motion
    def screw_to_matrix(screw_axis, theta):
        """Convert screw axis and angle to transformation matrix"""
        omega = screw_axis[:3]
        v = screw_axis[3:]

        so3_omega = vec_to_so3(omega)
        exp_so3 = matrix_exp3(so3_omega)

        # Calculate translation part
        if np.allclose(omega, 0):
            trans = v * theta
        else:
            omega_norm = np.linalg.norm(omega)
            omega_unit = omega / omega_norm
            so3_omega_unit = vec_to_so3(omega_unit)
            trans = np.dot(np.eye(3) * theta +
                          (1 - np.cos(theta * omega_norm)) / (omega_norm**2) * so3_omega_unit +
                          (theta * omega_norm - np.sin(theta * omega_norm)) / (omega_norm**3) *
                          np.dot(so3_omega_unit, so3_omega_unit), v)

        T = np.eye(4)
        T[:3, :3] = exp_so3
        T[:3, 3] = trans
        return T

    # 3. Forward kinematics example
    def forward_kinematics(joint_angles, screw_axes, M):
        """Compute forward kinematics"""
        T = M.copy()
        for i, (theta, screw_axis) in enumerate(zip(joint_angles, screw_axes)):
            exp_ti = screw_to_matrix(screw_axis, theta)
            T = np.dot(T, exp_ti)
        return T

    # Example usage
    # Define a simple 2-link planar arm
    joint_angles = [np.pi/4, np.pi/6]  # 45° and 30°

    # Screw axes for revolute joints (omega, v) - simplified example
    screw_axes = [
        np.array([0, 0, 1, 0, 0, 0]),  # First joint: rotation about z-axis
        np.array([0, 0, 1, 0, 0, 0])   # Second joint: rotation about z-axis
    ]

    # Initial transformation matrix
    M = np.eye(4)

    # Compute forward kinematics
    end_effector_pose = forward_kinematics(joint_angles, screw_axes, M)

    return {
        'end_effector_pose': end_effector_pose,
        'joint_angles': joint_angles
    }
```

## Audio Processing Libraries

### Speech Recognition

```python
import speech_recognition as sr
import pyttsx3
import pyaudio
import wave

# Speech recognition
def speech_to_text():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    try:
        # Use Google's speech recognition
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results: {e}")
        return None

# Text to speech
def text_to_speech(text, rate=200, volume=0.9):
    engine = pyttsx3.init()

    # Set properties
    engine.setProperty('rate', rate)  # Speed of speech
    engine.setProperty('volume', volume)  # Volume level (0.0 to 1.0)

    # Get available voices
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Change voice if needed

    # Speak the text
    engine.say(text)
    engine.runAndWait()

# Audio recording
def record_audio(filename, duration=5, sample_rate=44100, chunk_size=1024):
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )

    print("Recording...")
    frames = []

    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording")

    # Stop stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save to file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# Audio processing
def analyze_audio(filename):
    import librosa
    import numpy as np

    # Load audio
    y, sr = librosa.load(filename)

    # Extract features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return {
        'tempo': tempo,
        'beats': beats,
        'mfccs': mfccs,
        'spectral_centroids': spectral_centroids,
        'chroma': chroma,
        'duration': librosa.get_duration(y=y, sr=sr)
    }
```

### Vosk for Offline Speech Recognition

```python
# Note: Vosk needs to be installed separately
# pip install vosk

try:
    from vosk import Model, KaldiRecognizer
    import json
    import sys

    def vosk_offline_recognition(audio_file_path, model_path="model"):
        """Offline speech recognition using Vosk"""

        # Load model
        model = Model(model_path)

        # Create recognizer
        rec = KaldiRecognizer(model, 16000)  # Sample rate: 16kHz

        # Open audio file
        wf = open(audio_file_path, "rb")

        results = []

        while True:
            data = wf.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))

        # Final result
        results.append(json.loads(rec.FinalResult()))

        wf.close()

        # Extract text from results
        text_results = []
        for result in results:
            if 'text' in result and result['text']:
                text_results.append(result['text'])

        return ' '.join(text_results)

except ImportError:
    print("Vosk not available - install with: pip install vosk")

# Audio streaming with Vosk
def vosk_microphone_streaming(model_path="model"):
    """Real-time speech recognition from microphone"""
    import pyaudio

    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8000)

    print("Listening... Press Ctrl+C to stop")

    try:
        while True:
            data = stream.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = rec.Result()
                print(json.loads(result)['text'])
            else:
                # Partial result
                partial = rec.PartialResult()
                # Uncomment to see partial results
                # print(json.loads(partial)['partial'])
    except KeyboardInterrupt:
        print("\nStopped listening")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
```

## Data Processing Libraries

### Pandas

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# DataFrame creation and manipulation
def create_robot_dataframe():
    """Create a sample robot data DataFrame"""

    # Sample data
    timestamps = pd.date_range(start='2023-01-01', periods=100, freq='1S')
    positions_x = np.cumsum(np.random.normal(0, 0.1, 100)).cumsum()
    positions_y = np.cumsum(np.random.normal(0, 0.1, 100)).cumsum()
    velocities = np.random.normal(0.5, 0.2, 100)
    battery_levels = np.linspace(100, 20, 100)  # Decreasing battery

    df = pd.DataFrame({
        'timestamp': timestamps,
        'position_x': positions_x,
        'position_y': positions_y,
        'velocity': velocities,
        'battery_level': battery_levels,
        'status': np.random.choice(['active', 'charging', 'idle'], 100)
    })

    return df

def analyze_robot_data(df):
    """Analyze robot operational data"""

    # Basic statistics
    stats = df.describe()

    # Time-based analysis
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Group by status
    status_analysis = df.groupby('status').agg({
        'velocity': ['mean', 'std', 'min', 'max'],
        'battery_level': ['mean', 'min', 'max'],
        'position_x': ['mean', 'std'],
        'position_y': ['mean', 'std']
    }).round(3)

    # Velocity trends
    df['velocity_smoothed'] = df['velocity'].rolling(window=5).mean()

    # Battery consumption rate
    df['battery_consumption_rate'] = df['battery_level'].diff().abs() / df['timestamp'].diff().dt.total_seconds()

    return {
        'basic_stats': stats,
        'status_analysis': status_analysis,
        'smoothed_velocity': df['velocity_smoothed'],
        'consumption_rate': df['battery_consumption_rate'].mean()
    }

# Data cleaning and preprocessing
def clean_robot_data(df):
    """Clean and preprocess robot data"""

    # Remove duplicates
    df_clean = df.drop_duplicates(subset=['timestamp'])

    # Handle missing values
    df_clean = df_clean.fillna(method='forward')  # Forward fill
    df_clean = df_clean.fillna(0)  # Fill remaining with 0

    # Remove outliers using IQR method
    for col in ['position_x', 'position_y', 'velocity']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    # Normalize numerical columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numerical_cols = ['position_x', 'position_y', 'velocity', 'battery_level']
    df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

    return df_clean

# Export and import
def save_robot_data(df, filename):
    """Save robot data to various formats"""

    # Save as CSV
    df.to_csv(f"{filename}.csv", index=False)

    # Save as Parquet (more efficient for large datasets)
    df.to_parquet(f"{filename}.parquet", index=False)

    # Save as Excel
    df.to_excel(f"{filename}.xlsx", index=False)

    # Save specific columns
    df[['timestamp', 'position_x', 'position_y']].to_json(f"{filename}_positions.json", orient='records', date_format='iso')

def load_robot_data(filename):
    """Load robot data from various formats"""

    if filename.endswith('.csv'):
        return pd.read_csv(filename)
    elif filename.endswith('.parquet'):
        return pd.read_parquet(filename)
    elif filename.endswith('.xlsx'):
        return pd.read_excel(filename)
    elif filename.endswith('.json'):
        return pd.read_json(filename)
    else:
        raise ValueError("Unsupported file format")
```

### NumPy

```python
import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt

# Array creation and manipulation
def numpy_robotics_examples():
    """Examples of NumPy for robotics applications"""

    # Create transformation matrices
    def create_rotation_matrix(angle, axis='z'):
        """Create 2D or 3D rotation matrix"""
        if axis == 'z':
            return np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        elif axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])

    # Homogeneous transformation
    def create_transform(rotation_matrix, translation):
        """Create 4x4 homogeneous transformation matrix"""
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix[:3, :3]
        transform[:3, 3] = translation
        return transform

    # Vector operations
    def normalize_vector(v):
        """Normalize a vector"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def rotate_vector(vector, angle, axis='z'):
        """Rotate a vector by given angle around specified axis"""
        rot_matrix = create_rotation_matrix(angle, axis)
        # Only use 3x3 rotation part for vector rotation
        return np.dot(rot_matrix[:3, :3], vector)

    # Example usage
    angle = np.pi / 4  # 45 degrees
    rotation = create_rotation_matrix(angle, 'z')
    translation = np.array([1, 2, 0])
    transform = create_transform(rotation, translation)

    # Rotate a point
    point = np.array([1, 0, 0])
    rotated_point = rotate_vector(point, angle, 'z')

    return {
        'rotation_matrix': rotation,
        'transform_matrix': transform,
        'original_point': point,
        'rotated_point': rotated_point
    }

# Signal processing with NumPy/SciPy
def signal_processing_examples():
    """Signal processing examples for sensor data"""

    # Generate sample sensor data
    time = np.linspace(0, 10, 1000)
    # Simulate noisy sensor reading with trend
    signal_data = np.sin(2 * np.pi * 0.5 * time) + 0.1 * np.random.randn(len(time))
    signal_data += 0.02 * time  # Add linear trend

    # Apply filters
    # Moving average filter
    window_size = 50
    moving_avg = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')

    # Butterworth filter (using SciPy)
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * 100  # Assuming 100 Hz sampling
    low_cutoff = 5.0
    b, a = butter(3, low_cutoff / nyquist, btype='low')
    filtered_signal = filtfilt(b, a, signal_data)

    # Peak detection
    peaks = signal.find_peaks(filtered_signal, height=0.5, distance=50)[0]

    # Frequency analysis
    fft_result = np.fft.fft(filtered_signal)
    frequencies = np.fft.fftfreq(len(filtered_signal), 1/100)  # 100 Hz sampling
    magnitude_spectrum = np.abs(fft_result)

    return {
        'original_signal': signal_data,
        'filtered_signal': filtered_signal,
        'moving_average': moving_avg,
        'peaks': peaks,
        'frequencies': frequencies,
        'magnitude_spectrum': magnitude_spectrum
    }

# Mathematical operations
def mathematical_operations():
    """Common mathematical operations for robotics"""

    # Jacobian calculation example
    def calculate_jacobian(func, x, eps=1e-8):
        """Numerically calculate Jacobian of a function"""
        x = np.asarray(x)
        fx = func(x)
        jac = np.zeros((len(fx), len(x)))

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            fx_plus = func(x_plus)
            jac[:, i] = (fx_plus - fx) / eps

        return jac

    # Example function for Jacobian
    def robot_forward_kinematics(joint_angles):
        """Simple 2D planar robot forward kinematics"""
        l1, l2 = 1.0, 0.8  # Link lengths
        q1, q2 = joint_angles

        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

        return np.array([x, y])

    # Calculate Jacobian at specific joint angles
    joint_angles = np.array([np.pi/4, np.pi/6])
    jacobian = calculate_jacobian(robot_forward_kinematics, joint_angles)

    # Inverse kinematics (geometric solution for 2R robot)
    def inverse_kinematics_2R(x, y, l1=1.0, l2=0.8):
        """Geometric inverse kinematics for 2R planar robot"""
        r = np.sqrt(x**2 + y**2)

        # Check if position is reachable
        if r > l1 + l2:
            raise ValueError("Position is outside workspace")
        if r < abs(l1 - l2):
            raise ValueError("Position is inside workspace boundary")

        # Calculate elbow-up solution
        cos_q2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        sin_q2 = np.sqrt(1 - cos_q2**2)  # Take positive square root for elbow-up
        q2 = np.arctan2(sin_q2, cos_q2)

        k1 = l1 + l2 * cos_q2
        k2 = l2 * sin_q2
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return np.array([q1, q2])

    target_pos = np.array([1.2, 0.5])
    joint_solution = inverse_kinematics_2R(target_pos[0], target_pos[1])

    return {
        'jacobian': jacobian,
        'joint_solution': joint_solution,
        'target_position': target_pos
    }
```

## Visualization Libraries

### Matplotlib

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Robot trajectory visualization
def plot_robot_trajectory(positions, velocities=None, title="Robot Trajectory"):
    """Plot 2D or 3D robot trajectory"""

    fig, ax = plt.subplots(figsize=(10, 8))

    if positions.shape[1] == 2:  # 2D trajectory
        x, y = positions[:, 0], positions[:, 1]
        ax.plot(x, y, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(x[0], y[0], color='green', s=100, label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], color='red', s=100, label='End', zorder=5)

        # Plot velocity vectors if provided
        if velocities is not None:
            ax.quiver(x[::10], y[::10], velocities[::10, 0], velocities[::10, 1],
                     alpha=0.5, scale_units='xy', angles='xy', scale=1, color='orange')

    elif positions.shape[1] == 3:  # 3D trajectory
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End', zorder=5)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    if positions.shape[1] == 3:
        ax.set_zlabel('Z Position (m)')

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax

# Sensor data visualization
def plot_sensor_data(timestamps, sensor_data_dict, title="Sensor Data"):
    """Plot multiple sensor data streams"""

    n_sensors = len(sensor_data_dict)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 4*n_sensors), sharex=True)

    if n_sensors == 1:
        axes = [axes]

    for i, (sensor_name, data) in enumerate(sensor_data_dict.items()):
        axes[i].plot(timestamps, data, linewidth=1.5)
        axes[i].set_ylabel(sensor_name)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{sensor_name} Data')

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()

    return fig, axes

# Real-time plotting
def real_time_plot():
    """Create real-time plot for live sensor data"""

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    x_data, y_data = [], []

    # Create empty line object
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Real-time Data Stream')
    ax.grid(True, alpha=0.3)

    def animate(frame):
        # Generate new data point
        x_data.append(frame)
        y_data.append(np.sin(frame * 0.1) + np.random.normal(0, 0.1))

        # Keep only last 100 points
        if len(x_data) > 100:
            x_data.pop(0)
            y_data.pop(0)

        # Update plot
        line.set_data(x_data, y_data)

        # Adjust x-axis limits to show sliding window
        if len(x_data) > 0:
            ax.set_xlim(max(0, x_data[-1] - 100), x_data[-1])

        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, animate, interval=50, blit=True, cache_frame_data=False)

    return fig, ani

# Heatmap visualization for occupancy grids
def plot_occupancy_grid(grid, title="Occupancy Grid"):
    """Plot occupancy grid as heatmap"""

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(grid, cmap='RdYlBu_r', origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Occupancy Probability')

    ax.set_xlabel('X Cell Index')
    ax.set_ylabel('Y Cell Index')
    ax.set_title(title)

    return fig, ax

# 3D point cloud visualization
def plot_point_cloud(point_cloud, title="Point Cloud"):
    """Plot 3D point cloud data"""

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    if point_cloud.shape[1] >= 3:
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    return fig, ax
```

### Seaborn for Statistical Plots

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def statistical_robot_plots(df):
    """Create statistical plots for robot data analysis"""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=axes[0,0])
    axes[0,0].set_title('Feature Correlation Matrix')

    # 2. Pair plot for key variables
    key_vars = [col for col in numeric_cols if col in ['position_x', 'position_y', 'velocity', 'battery_level']]
    if len(key_vars) >= 2:
        sns.scatterplot(data=df, x=key_vars[0], y=key_vars[1], hue='status', ax=axes[0,1])
        axes[0,1].set_title(f'{key_vars[0]} vs {key_vars[1]}')

    # 3. Distribution plots
    for i, col in enumerate(key_vars[:2]):
        if i < 2:
            sns.histplot(data=df, x=col, hue='status', kde=True, ax=axes[1,i])
            axes[1,i].set_title(f'Distribution of {col}')

    # 4. Box plot for status comparison
    if 'status' in df.columns and len(key_vars) > 0:
        sns.boxplot(data=df, x='status', y=key_vars[0], ax=axes[1,1])
        axes[1,1].set_title(f'{key_vars[0]} by Status')
        axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig, axes

# Time series analysis plots
def time_series_plots(df):
    """Create time series plots for temporal analysis"""

    if 'timestamp' not in df.columns:
        print("No timestamp column found")
        return None, None

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Time series plot
    axes[0,0].plot(df['timestamp'], df['position_x'], label='X Position', alpha=0.7)
    axes[0,0].plot(df['timestamp'], df['position_y'], label='Y Position', alpha=0.7)
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Position (m)')
    axes[0,0].set_title('Position Over Time')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)

    # 2. Battery level over time
    axes[0,1].plot(df['timestamp'], df['battery_level'], 'g-', linewidth=2)
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Battery Level (%)')
    axes[0,1].set_title('Battery Level Over Time')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Velocity histogram
    axes[1,0].hist(df['velocity'], bins=30, alpha=0.7, color='orange')
    axes[1,0].set_xlabel('Velocity (m/s)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Velocity Distribution')
    axes[1,0].grid(True, alpha=0.3)

    # 4. 2D position scatter
    scatter = axes[1,1].scatter(df['position_x'], df['position_y'],
                               c=pd.to_numeric(pd.factorize(df['status'])[0]),
                               cmap='viridis', alpha=0.6)
    axes[1,1].set_xlabel('X Position (m)')
    axes[1,1].set_ylabel('Y Position (m)')
    axes[1,1].set_title('Position Scatter (Colored by Status)')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes
```

## Networking and Communication

### ZeroMQ

```python
import zmq
import json
import threading
import time

class ZMQRobotInterface:
    """ZeroMQ interface for robot communication"""

    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.port = port

    def start_server(self):
        """Start ZMQ server for robot commands"""
        socket = self.context.socket(zmq.REP)  # Reply socket
        socket.bind(f"tcp://*:{self.port}")

        print(f"Server listening on port {self.port}")

        try:
            while True:
                # Receive request
                message = socket.recv_string()
                print(f"Received request: {message}")

                # Parse command
                try:
                    cmd_data = json.loads(message)
                    response = self.process_command(cmd_data)
                except json.JSONDecodeError:
                    response = {"error": "Invalid JSON"}
                except Exception as e:
                    response = {"error": str(e)}

                # Send response
                socket.send_string(json.dumps(response))

        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            socket.close()

    def start_client(self):
        """Start ZMQ client to send commands"""
        socket = self.context.socket(zmq.REQ)  # Request socket
        socket.connect(f"tcp://localhost:{self.port}")

        return socket

    def send_command(self, command_data):
        """Send command to robot server"""
        socket = self.start_client()

        # Send command
        socket.send_string(json.dumps(command_data))

        # Receive response
        response = socket.recv_string()
        socket.close()

        return json.loads(response)

    def process_command(self, cmd_data):
        """Process robot command"""
        command = cmd_data.get('command')

        if command == 'move_to':
            x = cmd_data.get('x', 0)
            y = cmd_data.get('y', 0)
            # Simulate robot movement
            time.sleep(0.1)  # Simulate movement time
            return {"success": True, "message": f"Moved to ({x}, {y})"}

        elif command == 'get_position':
            # Simulate getting current position
            import random
            return {
                "success": True,
                "position": {
                    "x": random.uniform(-10, 10),
                    "y": random.uniform(-10, 10),
                    "theta": random.uniform(-3.14, 3.14)
                }
            }

        else:
            return {"success": False, "error": f"Unknown command: {command}"}

# Example usage
def zmq_example():
    # In one thread/process, start server
    def run_server():
        interface = ZMQRobotInterface(5555)
        interface.start_server()

    # In another thread, send commands
    def send_commands():
        interface = ZMQRobotInterface(5555)
        time.sleep(1)  # Wait for server to start

        # Send move command
        response = interface.send_command({
            "command": "move_to",
            "x": 1.0,
            "y": 2.0
        })
        print(f"Move response: {response}")

        # Get position
        response = interface.send_command({"command": "get_position"})
        print(f"Position response: {response}")

    # Run server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Send commands from main thread
    send_commands()
```

### WebSockets

```python
# Note: Requires installation of websockets library
# pip install websockets

try:
    import asyncio
    import websockets
    import json

    class WebSocketRobotServer:
        def __init__(self):
            self.clients = set()
            self.robot_state = {
                'position': {'x': 0, 'y': 0, 'theta': 0},
                'battery': 100,
                'status': 'idle'
            }

        async def register_client(self, websocket):
            self.clients.add(websocket)
            print(f"Client connected. Total clients: {len(self.clients)}")

        async def unregister_client(self, websocket):
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")

        async def broadcast_state(self):
            """Broadcast robot state to all connected clients"""
            if self.clients:
                message = json.dumps({
                    'type': 'state_update',
                    'data': self.robot_state
                })
                await asyncio.gather(
                    *[client.send(message) for client in self.clients],
                    return_exceptions=True
                )

        async def handle_command(self, websocket, command_data):
            """Handle incoming commands"""
            command = command_data.get('command')

            if command == 'move_to':
                x = command_data.get('x', self.robot_state['position']['x'])
                y = command_data.get('y', self.robot_state['position']['y'])

                # Update position (simulate movement)
                self.robot_state['position']['x'] = x
                self.robot_state['position']['y'] = y

                response = {
                    'type': 'command_response',
                    'success': True,
                    'message': f'Moved to ({x}, {y})'
                }

            elif command == 'get_state':
                response = {
                    'type': 'state_response',
                    'data': self.robot_state
                }

            else:
                response = {
                    'type': 'command_response',
                    'success': False,
                    'error': f'Unknown command: {command}'
                }

            await websocket.send(json.dumps(response))
            await self.broadcast_state()

        async def handler(self, websocket, path):
            await self.register_client(websocket)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_command(websocket, data)
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON'
                        }))
            finally:
                await self.unregister_client(websocket)

        async def start_server(self, host='localhost', port=8765):
            server = await websockets.serve(self.handler, host, port)
            print(f"WebSocket server started on ws://{host}:{port}")
            await server.wait_closed()

    # Example client function
    async def websocket_client_example():
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            # Send move command
            command = {
                "command": "move_to",
                "x": 1.0,
                "y": 2.0
            }
            await websocket.send(json.dumps(command))

            response = await websocket.recv()
            print(f"Response: {response}")

            # Listen for state updates
            try:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data['type'] == 'state_update':
                        print(f"State updated: {data['data']}")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")

except ImportError:
    print("websockets library not available")
```

## System and Hardware Libraries

### Serial Communication

```python
import serial
import serial.tools.list_ports
import time

class SerialRobotInterface:
    """Serial interface for robot communication"""

    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None

    def list_serial_ports(self):
        """List available serial ports"""
        ports = serial.tools.list_ports.comports()
        available_ports = []

        for port in ports:
            available_ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            })

        return available_ports

    def connect(self, port=None):
        """Connect to serial port"""
        if port:
            self.port = port

        if not self.port:
            available_ports = self.list_serial_ports()
            if available_ports:
                self.port = available_ports[0]['device']  # Use first available
                print(f"Using port: {self.port}")
            else:
                raise Exception("No serial ports available")

        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from serial port"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")

    def send_command(self, command, terminator='\n'):
        """Send command to robot"""
        if not self.serial_conn or not self.serial_conn.is_open:
            raise Exception("Not connected to serial port")

        full_command = command + terminator
        self.serial_conn.write(full_command.encode())
        print(f"Sent: {full_command.strip()}")

    def read_response(self, timeout=2):
        """Read response from robot"""
        if not self.serial_conn or not self.serial_conn.is_open:
            raise Exception("Not connected to serial port")

        start_time = time.time()
        response_lines = []

        while time.time() - start_time < timeout:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode().strip()
                if line:
                    response_lines.append(line)
                    # Break if we get an acknowledgment or end marker
                    if line.startswith('OK') or line.startswith('DONE'):
                        break
            time.sleep(0.01)

        return '\n'.join(response_lines)

    def send_and_receive(self, command, timeout=2):
        """Send command and wait for response"""
        self.send_command(command)
        return self.read_response(timeout)

    def ping(self):
        """Ping robot to check connection"""
        response = self.send_and_receive('PING', timeout=1)
        return 'PONG' in response or 'OK' in response

# Example usage
def serial_example():
    robot = SerialRobotInterface()

    # List available ports
    ports = robot.list_serial_ports()
    print("Available ports:")
    for port in ports:
        print(f"  {port['device']}: {port['description']}")

    # Connect to robot
    if robot.connect():
        # Test connection
        if robot.ping():
            print("Robot responded to ping!")

            # Send some commands
            response = robot.send_and_receive('GET_POS')
            print(f"Position: {response}")

            # Move robot
            robot.send_command('MOVE_TO 100 200')
            time.sleep(2)  # Wait for movement

        robot.disconnect()
```

### GPIO (for Raspberry Pi)

```python
# Note: GPIO libraries are hardware-specific
# For Raspberry Pi, you might use RPi.GPIO or gpiozero

try:
    import RPi.GPIO as GPIO
    import time

    class GPIORobotControl:
        def __init__(self, pins_config):
            """
            pins_config example:
            {
                'motor_left': {'dir': 18, 'pwm': 19},
                'motor_right': {'dir': 23, 'pwm': 24},
                'servo': 12,
                'sensors': {'ultrasonic': {'trig': 20, 'echo': 21}}
            }
            """
            self.pins = pins_config
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Setup motor pins
            if 'motor_left' in self.pins:
                left_pins = self.pins['motor_left']
                GPIO.setup(left_pins['dir'], GPIO.OUT)
                GPIO.setup(left_pins['pwm'], GPIO.OUT)
                self.left_pwm = GPIO.PWM(left_pins['pwm'], 1000)  # 1kHz PWM
                self.left_pwm.start(0)

            if 'motor_right' in self.pins:
                right_pins = self.pins['motor_right']
                GPIO.setup(right_pins['dir'], GPIO.OUT)
                GPIO.setup(right_pins['pwm'], GPIO.OUT)
                self.right_pwm = GPIO.PWM(right_pins['pwm'], 1000)
                self.right_pwm.start(0)

        def set_motor_speed(self, motor, speed, direction=1):
            """
            Set motor speed and direction
            motor: 'left' or 'right'
            speed: 0-100 percentage
            direction: 1 for forward, -1 for backward
            """
            if motor == 'left':
                pwm = self.left_pwm
                dir_pin = self.pins['motor_left']['dir']
            elif motor == 'right':
                pwm = self.right_pwm
                dir_pin = self.pins['motor_right']['dir']
            else:
                raise ValueError("Motor must be 'left' or 'right'")

            # Set direction
            GPIO.output(dir_pin, GPIO.HIGH if direction >= 0 else GPIO.LOW)

            # Set speed
            pwm.ChangeDutyCycle(abs(speed))

        def stop_motors(self):
            """Stop both motors"""
            if hasattr(self, 'left_pwm'):
                self.left_pwm.ChangeDutyCycle(0)
            if hasattr(self, 'right_pwm'):
                self.right_pwm.ChangeDutyCycle(0)

        def cleanup(self):
            """Clean up GPIO"""
            self.stop_motors()
            GPIO.cleanup()

    # Alternative using gpiozero (often easier)
    from gpiozero import Motor, Servo, DistanceSensor
    import time

    class GPIOZeroRobot:
        def __init__(self, left_motor_pins=(18, 19), right_motor_pins=(23, 24), servo_pin=12):
            self.left_motor = Motor(forward=left_motor_pins[0], backward=left_motor_pins[1])
            self.right_motor = Motor(forward=right_motor_pins[0], backward=right_motor_pins[1])
            self.servo = Servo(servo_pin)

            # Ultrasonic sensor
            self.distance_sensor = DistanceSensor(echo=21, trigger=20)

        def move_forward(self, speed=0.5):
            self.left_motor.forward(speed)
            self.right_motor.forward(speed)

        def move_backward(self, speed=0.5):
            self.left_motor.backward(speed)
            self.right_motor.backward(speed)

        def turn_left(self, speed=0.5):
            self.left_motor.backward(speed)
            self.right_motor.forward(speed)

        def turn_right(self, speed=0.5):
            self.left_motor.forward(speed)
            self.right_motor.backward(speed)

        def stop(self):
            self.left_motor.stop()
            self.right_motor.stop()

        def get_distance(self):
            return self.distance_sensor.distance

        def set_servo_position(self, position):
            """Position between -1 and 1"""
            self.servo.value = position

    def gpio_example():
        # Initialize robot
        robot = GPIOZeroRobot()

        # Move forward
        robot.move_forward(0.5)
        time.sleep(2)

        # Stop
        robot.stop()
        time.sleep(1)

        # Check distance
        distance = robot.get_distance()
        print(f"Distance: {distance:.2f} m")

        # Clean up
        robot.stop()

except ImportError:
    print("GPIO libraries not available (requires Raspberry Pi hardware)")
```

## Utility Libraries

### Configuration Management

```python
import json
import yaml
import configparser
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os

@dataclass
class RobotConfig:
    """Data class for robot configuration"""
    name: str = "default_robot"
    max_speed: float = 1.0
    acceleration: float = 0.5
    operating_mode: str = "autonomous"
    sensors: Dict[str, Any] = None

    def __post_init__(self):
        if self.sensors is None:
            self.sensors = {
                "camera": {"enabled": True, "resolution": [640, 480]},
                "lidar": {"enabled": True, "range": 10.0},
                "imu": {"enabled": True}
            }

class ConfigManager:
    """Configuration manager for robot systems"""

    def __init__(self, config_path: str = "config"):
        self.config_path = config_path
        self.config = RobotConfig()

    def load_json(self, filepath: str) -> RobotConfig:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create config object from data
        config = RobotConfig(**data)
        self.config = config
        return config

    def save_json(self, filepath: str, config: RobotConfig = None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config

        with open(filepath, 'w') as f:
            json.dump(asdict(config), f, indent=2)

    def load_yaml(self, filepath: str) -> RobotConfig:
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        config = RobotConfig(**data)
        self.config = config
        return config

    def save_yaml(self, filepath: str, config: RobotConfig = None):
        """Save configuration to YAML file"""
        if config is None:
            config = self.config

        with open(filepath, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, indent=2)

    def load_ini(self, filepath: str) -> RobotConfig:
        """Load configuration from INI file"""
        parser = configparser.ConfigParser()
        parser.read(filepath)

        # Convert INI structure to config dict
        config_dict = {}

        # Robot section
        if 'ROBOT' in parser:
            robot_section = parser['ROBOT']
            config_dict['name'] = robot_section.get('name', 'default_robot')
            config_dict['max_speed'] = robot_section.getfloat('max_speed', 1.0)
            config_dict['acceleration'] = robot_section.getfloat('acceleration', 0.5)
            config_dict['operating_mode'] = robot_section.get('operating_mode', 'autonomous')

        # Sensors section
        if 'SENSORS' in parser:
            sensors_section = parser['SENSORS']
            sensors = {}
            for key, value in sensors_section.items():
                # Parse sensor configurations
                sensors[key] = json.loads(value) if value.startswith('{') else value
            config_dict['sensors'] = sensors

        config = RobotConfig(**config_dict)
        self.config = config
        return config

    def save_ini(self, filepath: str, config: RobotConfig = None):
        """Save configuration to INI file"""
        if config is None:
            config = self.config

        parser = configparser.ConfigParser()

        # Robot section
        parser['ROBOT'] = {
            'name': config.name,
            'max_speed': str(config.max_speed),
            'acceleration': str(config.acceleration),
            'operating_mode': config.operating_mode
        }

        # Sensors section
        parser['SENSORS'] = {}
        for sensor_name, sensor_config in config.sensors.items():
            parser['SENSORS'][sensor_name] = json.dumps(sensor_config)

        with open(filepath, 'w') as f:
            parser.write(f)

    def get_config(self) -> RobotConfig:
        """Get current configuration"""
        return self.config

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        config_dict = asdict(self.config)
        config_dict.update(kwargs)
        self.config = RobotConfig(**config_dict)

# Environment variable configuration
def load_config_from_env():
    """Load configuration from environment variables"""
    config = RobotConfig()

    # Update from environment variables
    config.name = os.getenv('ROBOT_NAME', config.name)
    config.max_speed = float(os.getenv('MAX_SPEED', config.max_speed))
    config.acceleration = float(os.getenv('ACCELERATION', config.acceleration))
    config.operating_mode = os.getenv('OPERATING_MODE', config.operating_mode)

    return config

# Example configuration files

# config.json
json_config = """
{
  "name": "delivery_robot_01",
  "max_speed": 0.8,
  "acceleration": 0.3,
  "operating_mode": "delivery",
  "sensors": {
    "camera": {
      "enabled": true,
      "resolution": [1280, 720],
      "fov": 60
    },
    "lidar": {
      "enabled": true,
      "range": 20.0,
      "resolution": 0.5
    },
    "imu": {
      "enabled": true,
      "rate": 100
    }
  }
}
"""

# config.yaml
yaml_config = """
name: "inspection_robot_01"
max_speed: 0.5
acceleration: 0.2
operating_mode: "inspection"
sensors:
  camera:
    enabled: true
    resolution: [640, 480]
    fov: 90
  lidar:
    enabled: true
    range: 15.0
    resolution: 1.0
  imu:
    enabled: true
    rate: 50
"""
```

### Logging and Monitoring

```python
import logging
import logging.handlers
import sys
from datetime import datetime
import os

class RobotLogger:
    """Advanced logging for robot systems"""

    def __init__(self, name="RobotSystem", log_dir="logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # File handler - daily rotating
        log_file = os.path.join(log_dir, f"{name.lower()}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=7
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Error file handler - only errors
        error_file = os.path.join(log_dir, f"{name.lower()}_errors.log")
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)

        # Prevent duplicate logs if parent logger exists
        self.logger.propagate = False

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def log_robot_state(self, state_data):
        """Log robot state information"""
        state_str = ", ".join([f"{k}={v}" for k, v in state_data.items()])
        self.info(f"Robot state: {state_str}")

    def log_sensor_data(self, sensor_name, value):
        """Log sensor data with timestamp"""
        self.debug(f"Sensor[{sensor_name}]: {value}")

    def log_navigation_event(self, event_type, details):
        """Log navigation-related events"""
        self.info(f"Navigation {event_type}: {details}")

# Performance monitoring
class PerformanceMonitor:
    """Monitor performance metrics"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, name):
        """Start a timer"""
        import time
        self.start_times[name] = time.time()

    def stop_timer(self, name):
        """Stop a timer and record duration"""
        import time
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            return duration
        return None

    def get_average_time(self, name):
        """Get average execution time for a named operation"""
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0

    def get_metric_summary(self):
        """Get summary of all metrics"""
        summary = {}
        for name, times in self.metrics.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times)
                }
        return summary

# Example usage
def logging_example():
    # Create logger
    robot_logger = RobotLogger("DeliveryRobot")

    # Log some events
    robot_logger.info("Robot initialized")
    robot_logger.log_robot_state({
        'position': (1.0, 2.0, 0.5),
        'battery': 85.3,
        'status': 'ready'
    })

    # Performance monitoring
    perf_monitor = PerformanceMonitor()

    # Monitor a function
    perf_monitor.start_timer('navigation_calculation')
    # ... some computation ...
    import time
    time.sleep(0.1)  # Simulate work
    duration = perf_monitor.stop_timer('navigation_calculation')

    robot_logger.info(f"Navigation calculation took {duration:.3f}s")

    # Print performance summary
    summary = perf_monitor.get_metric_summary()
    for name, stats in summary.items():
        robot_logger.info(f"{name}: avg={stats['average']:.3f}s, count={stats['count']}")
```

## Installation and Setup

### Requirements Management

```bash
# requirements.txt for Physical AI projects
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
opencv-python>=4.5.0
Pillow>=8.3.0
torch>=1.10.0
torchvision>=0.11.0
transformers>=4.12.0
scikit-learn>=1.0.0
pandas>=1.3.0
pyserial>=3.5
speechrecognition>=3.8.1
pyttsx3>=2.90
rclpy>=3.1.0
ros2
websockets>=10.0
pyyaml>=6.0
requests>=2.26.0
seaborn>=0.11.0
librosa>=0.9.0
vosk>=0.3.45
```

### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv physical_ai_env

# Activate environment
source physical_ai_env/bin/activate  # Linux/Mac
# physical_ai_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install ROS 2 Python packages
pip install -r requirements_ros.txt
```

### Docker Setup for Isolated Environments

```dockerfile
# Dockerfile for Physical AI development
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    locales \
    software-properties-common \
    && locale-gen en_US en_US.UTF-8 \
    && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-desktop \
    && apt-get install -y python3-colcon-common-extensions \
    && apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential \
    && rosdep init \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS 2 environment
ENV ROS_DISTRO=humble
ENV ROS_ROOT=/opt/ros/humble
ENV PATH=$ROS_ROOT/bin:$PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3.10/site-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Set up workspace
RUN mkdir -p /workspace/src
WORKDIR /workspace

# Source ROS 2 setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
```

## Best Practices

### 1. Error Handling and Logging

```python
import logging
from functools import wraps

def robust_function(func):
    """Decorator for robust function execution with error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            # Return safe default or re-raise based on context
            return None
    return wrapper

@robust_function
def sensor_reading_function():
    # Robust sensor reading code
    pass
```

### 2. Resource Management

```python
class ResourceManager:
    """Manage system resources for robot operations"""

    def __enter__(self):
        # Acquire resources
        self.acquire_resources()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release resources
        self.release_resources()

    def acquire_resources(self):
        # Open files, connect to devices, etc.
        pass

    def release_resources(self):
        # Close files, disconnect devices, etc.
        pass

# Usage
with ResourceManager() as rm:
    # Use resources
    pass
```

### 3. Configuration Validation

```python
from typing import Dict, Any
import json

def validate_robot_config(config: Dict[str, Any]) -> bool:
    """Validate robot configuration"""
    required_fields = ['name', 'max_speed', 'sensors']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Validate sensor configurations
    for sensor_name, sensor_config in config['sensors'].items():
        if not isinstance(sensor_config, dict):
            raise ValueError(f"Sensor {sensor_name} configuration must be a dict")

    # Validate numeric ranges
    if config['max_speed'] <= 0 or config['max_speed'] > 5.0:
        raise ValueError("max_speed must be between 0 and 5.0")

    return True
```

This comprehensive reference covers all the essential Python libraries needed for Physical AI and robotics development. Each section includes practical examples and best practices for real-world applications.