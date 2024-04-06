import os

def convert_filenames_to_lowercase(directory):
    for filename in os.listdir(directory):
        # Construct the old and new file paths
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, filename.lower())

        # Rename only if the old and new names are different 
        if old_filepath != new_filepath:
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {filename.lower()}")

if __name__ == "__main__":
    folder_path = "kilian/alignments_hhr/test_yCD/downloaded_cif_files"  # Replace with the actual path to your folder
    convert_filenames_to_lowercase(folder_path)
