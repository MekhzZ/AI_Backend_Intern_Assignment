import cv2
import os

# Step 1: Specify the directories
input_directory = './assets/rawExcelExtractedVideo'  # Replace with your source directory
output_directory = './assets/croppedVideo'  # Replace with your target directory

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Step 2: Define the cropping function
def crop_video(input_path, output_path, crop_size=(224, 224)):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    
    # Calculate cropping coordinates for the center
    crop_width, crop_height = crop_size
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    
    # Define the VideoWriter for the output video
    out = cv2.VideoWriter(output_path, fourcc, fps, crop_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Crop the frame
        cropped_frame = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
        
        # Write the cropped frame to the output video
        out.write(cropped_frame)
    
    # Release resources
    cap.release()
    out.release()

# Step 3: Process all videos in the input directory
for filename in os.listdir(input_directory):
    input_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)
    
    if os.path.isfile(input_path):
        print(f"Processing {filename}...")
        crop_video(input_path, output_path)
        print(f"Cropped video saved to {output_path}")
