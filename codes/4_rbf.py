import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Mediapipe Face Detection setup
mp_face_detection = mp.solutions.face_detection

def extract_face_detection_features(video_path):
    """
    Extracts face detection features (bounding box and confidence score) for each frame in the video.
    Returns a list of features.
    """
    cap = cv2.VideoCapture(video_path)
    features_list = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face
            results = face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box and confidence
                    bboxC = detection.location_data.relative_bounding_box
                    confidence = detection.score[0]
                    features = [
                        bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height, confidence
                    ]
                    features_list.append(features)
            else:
                # Append None if no face is detected
                features_list.append(None)

    cap.release()
    return features_list

def preprocess_and_label_videos(video_paths, labels):
    """
    Extracts features from videos, assigns labels, and prepares data for training.
    """
    data = []
    label_list = []

    for video_path, label in zip(video_paths, labels):
        print(f"Processing video: {video_path} with label: {label}")
        features = extract_face_detection_features(video_path)
        for feature in features:
            if feature is not None:
                data.append(feature)
                label_list.append(label)

    return np.array(data), np.array(label_list)

# Videos and labels
video_paths = ["./assets/Datasets/diljith.mp4", "./assets/Datasets/messi.mp4", "./assets/Datasets/ronaldo.mp4"]  # Replace with your paths
labels = ["Diljith", "Messi", "Ronaldo"]

# Extract features and labels
X, y = preprocess_and_label_videos(video_paths, labels)

# Reshape features to ensure compatibility
X = np.reshape(X, (X.shape[0], -1))

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

# Define the SVM model with RBF kernel
svm_model_rbf = SVC(kernel="rbf", random_state=42)

# Perform Grid Search for Hyperparameter Tuning
param_grid = {
    "C": [0.1, 1, 10, 100],  # Regularization parameter
    "gamma": [1, 0.1, 0.01, 0.001]  # Kernel coefficient
}

grid_search = GridSearchCV(estimator=svm_model_rbf, param_grid=param_grid, cv=5, scoring="accuracy", verbose=2)
grid_search.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:", grid_search.best_params_)

# Train with best parameters
best_rbf_model = grid_search.best_estimator_
best_rbf_model.fit(X_train, y_train)

# Evaluate model
y_pred = best_rbf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"RBF Kernel Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and preprocessing objects
save_dir = "models_face_detection_rbf"
os.makedirs(save_dir, exist_ok=True)

# Save SVM model
with open(os.path.join(save_dir, "svm_face_detection_rbf_model.pkl"), "wb") as model_file:
    pickle.dump(best_rbf_model, model_file)

# Save scaler
with open(os.path.join(save_dir, "scaler.pkl"), "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save label encoder
with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)

print("RBF Model, scaler, and label encoder saved successfully!")
