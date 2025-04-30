import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# ✅ Load emotion detection model
model = load_model('C:/Users/hp/Desktop/python projects/expression+age+gender/model_file_30epochs.h5')

# ✅ Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# ✅ Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if frame not captured

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y + h, x:x + w]

        if face_roi.size == 0:
            continue  # Skip empty detections

        # ✅ Process for emotion detection
        resized = cv2.resize(face_roi, (48, 48))  # Resize to 48x48 for model input
        normalized = resized / 255.0  # Normalize pixel values
        reshaped = np.reshape(normalized, (1, 48, 48, 1))  # Reshape for model

        # ✅ Predict emotion
        emotion_prediction = model.predict(reshaped)
        emotion_label = labels_dict[np.argmax(emotion_prediction, axis=1)[0]]

        # ✅ Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ✅ Show output frame
    cv2.imshow("Facial Expression Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
