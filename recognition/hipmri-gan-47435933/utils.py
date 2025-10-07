import os
import zipfile

# Path to your keras_slices_data folder
base_dir = r"C:\Users\samue\Documents\Semester 2 2025\COMP3710\PatternAnalysis-2025\recognition\keras_slices_data"

# Loop over all subfolders (train/test/seg_train/seg_test)
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"Unzipping folder: {folder}")
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(folder_path, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract into the same folder
                zip_ref.extractall(folder_path)

    print(f"Finished unzipping {folder}")

