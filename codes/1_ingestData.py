import os
import pandas as pd
import requests

# Step 1: Load Excel File
excel_file = './assets/excelFile/assignmentData.xlsx'  # Replace with your Excel file path
df = pd.read_excel(excel_file)

# Step 2: Specify Destination Directory
destination_dir = './assets/rawExcelExtractedVideo'  # Replace with your desired directory

# Create the directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Step 3: Download Videos
for index, row in df.iterrows():
    video_url = row['Video URL']  # Column name for video URLs
    
    # Get the video name from the URL
    video_name = os.path.basename(video_url)
    save_path = os.path.join(destination_dir, video_name)
    
    try:
        # Download video
        print(f"Downloading {video_name}...")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        # Save video to destination directory
        with open(save_path, 'wb') as video_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Filter out keep-alive new chunks
                    video_file.write(chunk)
        print(f"Downloaded and saved as: {save_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {video_url}. Error: {e}")
