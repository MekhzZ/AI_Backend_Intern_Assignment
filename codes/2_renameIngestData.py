import os

# Step 1: Specify the directory containing the downloaded videos
directory = './assets/rawExcelExtractedVideo'  # Replace with your directory path

# Step 2: Rename files
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    # Check if it's a file and does not already have a .mp4 extension
    if os.path.isfile(file_path) and not filename.endswith('.mp4'):
        # Add .mp4 extension
        new_file_path = os.path.join(directory, filename + '.mp4')
        os.rename(file_path, new_file_path)
        print(f"Renamed: {file_path} -> {new_file_path}")
