# QuantumSaboteur/utils/data_loader.py
import numpy as np
import os
from typing import Optional # For type hinting

def load_txt_data(file_path: str, delimiter: str = None, dtype: type = float) -> Optional[np.ndarray]:
    """
    Loads numerical data from a text file using NumPy.

    Args:
        file_path (str): The full path to the text file.
        delimiter (str, optional): The string used to separate values.
                                   Defaults to None (handles whitespace).
        dtype (type, optional): The data type of the resulting array.
                                Defaults to float. For one-hot encoded labels
                                like '0.0 1.0' from your example, loading as
                                float is appropriate and avoids DeprecationWarnings
                                that can occur if dtype=int is forced on
                                float-like text.

    Returns:
        np.ndarray: A NumPy array containing the loaded data.
                    Returns None if the file is not found or an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        # Load data using the specified dtype.
        # If loading integer labels that are represented as floats in the file (e.g., "1.0"),
        # it's best to load as float and then use .astype(int) in the calling code if needed.
        data = np.loadtxt(file_path, delimiter=delimiter, dtype=dtype)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# if __name__ == '__main__':
#     # This example block is for testing data_loader.py directly.
#     # It assumes the following directory structure:
#     # QuantumSaboteur/
#     # ├── utils/               <-- This script (data_loader.py) is here
#     # │   └── data_loader.py
#     # ├── experiment_data/     <-- Data is here (sibling to utils/)
#     # │   └── bank/
#     # │       ├── x_train.txt
#     # │       └── y_train.txt
#     # └── main.py              (or other scripts at the project root)

#     # To construct the path to 'experiment_data' from 'utils', we go up one level.
#     # This relative path works if you run `python data_loader.py` from within the `utils` directory.
#     base_data_path = "../experiment_data"
    
#     # If you were running a script from the 'QuantumSaboteur' root (like main.py),
#     # the base_data_path to 'experiment_data' would simply be "experiment_data".

#     print(f"Calculated base data path for testing (relative to utils/): {os.path.abspath(base_data_path)}")

#     # --- Test loading x_train.txt (features, typically float) ---
#     x_train_file = os.path.join(base_data_path, "bank", "x_train.txt")
#     print(f"\nAttempting to load x_train from: {os.path.abspath(x_train_file)}")
#     x_train_data = load_txt_data(x_train_file) # Default dtype is float
#     if x_train_data is not None:
#         print(f"Successfully loaded x_train_data. Shape: {x_train_data.shape}")
#         if x_train_data.size > 0:
#             print(f"First row of x_train_data: {x_train_data[0]}")
#         else:
#             print("x_train_data is empty.")

#     # --- Test loading y_train.txt (one-hot encoded labels, e.g., "0.0 1.0") ---
#     y_train_file = os.path.join(base_data_path, "bank", "y_train.txt")
#     print(f"\nAttempting to load y_train from: {os.path.abspath(y_train_file)}")
#     # Load as float, as the file contains float representations like "1.000e+00"
#     y_train_data = load_txt_data(y_train_file, dtype=float)
#     if y_train_data is not None:
#         print(f"Successfully loaded y_train_data (as float). Shape: {y_train_data.shape}")
#         if y_train_data.size > 0:
#             print(f"First row of y_train_data: {y_train_data[0]}") # Should show [0. 1.] or similar
#         else:
#             print("y_train_data is empty.")

#         # Example: If you need to convert these one-hot encoded labels to single integer class indices
#         if y_train_data.ndim == 2 and y_train_data.shape[1] > 1: # Check if it looks like one-hot
#             try:
#                 y_train_class_indices = np.argmax(y_train_data, axis=1)
#                 print(f"Converted y_train_data to class indices. Shape: {y_train_class_indices.shape}")
#                 if y_train_class_indices.size > 0:
#                     print(f"First 5 class indices: {y_train_class_indices[:5]}")
#             except Exception as e:
#                 print(f"Could not convert y_train_data to class indices using argmax: {e}")
#         elif y_train_data.ndim == 1:
#             print("y_train_data appears to be already single-column (possibly class indices or regression targets).")
#         else:
#              print("y_train_data is not in a typical one-hot encoded format for argmax conversion along axis=1.")


#     # --- Test loading a non-existent file ---
#     non_existent_file = os.path.join(base_data_path, "bank", "non_existent.txt")
#     print(f"\nAttempting to load a non-existent file from: {os.path.abspath(non_existent_file)}")
#     non_existent_data = load_txt_data(non_existent_file)
#     if non_existent_data is None:
#         print("Correctly handled non-existent file.")
