import cv2
import pickle
import numpy as np
import mediapipe as mp

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

def predict_faces_in_video(video_path, output_path=None):
    """
    Predicts the label of the face detected in each frame of a video.
    Processes only faces with confidence > 80%.
    Collects and prints all unique predictions for the entire video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Define video writer if output is required
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set to store unique predictions
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

                    # Filter detections based on confidence
                    if confidence > 0.8:  # Only process detections with confidence > 80%
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

                            # Draw bounding box and label
                            h, w, _ = frame.shape
                            xmin = int(bboxC.xmin * w)
                            ymin = int(bboxC.ymin * h)
                            width = int(bboxC.width * w)
                            height = int(bboxC.height * h)
                            cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
                            cv2.putText(frame, f"{prediction} ({confidence:.2f})", (xmin, ymin - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error in prediction: {e}")

            # Display the frame
            cv2.imshow("Video", frame)

            # Save the frame with predictions if output is required
            if out:
                out.write(frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Print unique predictions at the end of the video
    if unique_predictions:
        print("Predicted labels for the video:", unique_predictions)
    else:
        print("No faces with confidence > 80% were detected.")

# Example usage
video_path = "D:/AI_Backend_Intern_Assignment/assets/rawExcelExtractedVideo/hd-429145826826558.mp4"  # Replace with your input video path
output_path = "./models/output.mp4"  # Replace with your desired output video path
predict_faces_in_video(video_path, output_path)
