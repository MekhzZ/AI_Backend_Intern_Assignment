import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import pandas as pd

# Load the trained model, scaler, and label encoder
save_dir = "models_face_detection"  # Replace with your model directory
try:
    with open(f"{save_dir}/svm_face_detection_rbf_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open(f"{save_dir}/scaler.pkl", "rb") as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    with open(f"{save_dir}/label_encoder.pkl", "rb") as label_encoder_file:
        loaded_label_encoder = pickle.load(label_encoder_file)
except Exception as e:
    print(f"Error loading model components: {e}")
    exit()

# Mediapipe Face Detection setup
mp_face_detection = mp.solutions.face_detection

def predict_video_labels(video_path):
    """
    Predicts the unique labels for a given video.
    Returns a set of unique labels for the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return set()

    unique_predictions = set()

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
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

                    # Process only detections with confidence > 80%
                    if confidence > 0.8:
                        features = np.array([
                            bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height, confidence
                        ])

                        # Reshape the features for compatibility
                        features_reshaped = features.reshape(1, -1)

                        # Normalize features using the saved scaler
                        try:
                            features_normalized = loaded_scaler.transform(features_reshaped)

                            # Predict using the loaded model
                            y_pred_numeric = loaded_model.predict(features_normalized)

                            # Decode label
                            prediction = loaded_label_encoder.inverse_transform(y_pred_numeric)[0]

                            # Add prediction to set
                            unique_predictions.add(prediction)
                        except Exception as e:
                            print(f"Error in prediction: {e}")

    cap.release()
    return unique_predictions

def process_all_videos_in_folder(folder_path, output_excel_path):
    """
    Processes all video files in a folder, predicts labels for each video, and saves the results in an Excel file.
    """
    # Supported video file extensions
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv")

    # List to store results
    results = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(supported_extensions):
            video_path = os.path.join(folder_path, file_name)
            print(f"Processing: {file_name}")

            # Predict labels for the video
            labels = predict_video_labels(video_path)

            # Append results (video file name and labels)
            results.append({
                "Video File": file_name,
                "Predicted Labels": ", ".join(labels) if labels else "No Face Detected"
            })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save DataFrame to Excel
    df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")

# Example usage
folder_path = "D:/AI_Backend_Intern_Assignment/assets/rawExcelExtractedVideo"  # Replace with your folder path
output_excel_path = "video_predictions.xlsx"  # Desired Excel output file
process_all_videos_in_folder(folder_path, output_excel_path)
