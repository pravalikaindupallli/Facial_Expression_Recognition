import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set dataset path
dataset_path = r"C:/Users/hp/Desktop/python projects/expression+age+gender/archive (5)\UTKFace"  # Change path accordingly

# Define input image size
IMG_SIZE = 200

# Initialize lists to store data
images, ages, genders = [], [], []

# Read dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):  # Ensure it's an image file
        try:
            # Extract labels (Format: age_gender_ethnicity.jpg)
            parts = filename.split("_")
            age = int(parts[0])  # First part is the age
            gender = int(parts[1])  # Second part is gender (0 = male, 1 = female)

            # Read and resize image
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize pixel values (0-1)

            # Append to lists
            images.append(img)
            ages.append(age)
            genders.append(gender)

        except Exception as e:
            print("Error loading image:", filename, e)

# Convert lists to numpy arrays
images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)

# Normalize ages (age range: 0-100 → scale to 0-1)
ages = ages / 100.0

# Convert gender to categorical (one-hot encoding: Male=0, Female=1)
genders = to_categorical(genders, num_classes=2)

# Split data into train & validation sets
X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
    images, ages, genders, test_size=0.2, random_state=42
)
