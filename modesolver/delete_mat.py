import os
import glob

from pathlib import Path

def delete_mat_files(folder_path):
    """
    Recursively delete all .mat files whose immediate parent folder is named 'modes'.
    """
    base = Path(folder_path)

    # Search for all .mat files under base
    for mat_file in base.rglob("*.mat"):
        # Check if the file's parent directory is named 'modes'
        if mat_file.parent.name == "modes":
            try:
                mat_file.unlink()
                print(f"Deleted: {mat_file}")
            except:
                pass

if __name__ == "__main__":
    # Replace this with the path to the folder you want to scan
    folder = Path("pysvmodes")
    delete_mat_files(folder)
