# utils/file_utils.py
import os
import pandas as pd

def ensure_dir(directory):
    """Ensure that the directory exists, if directory is not empty."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_csv(df, file_path):
    """Save a pandas DataFrame to CSV, ensuring that the directory exists."""
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
    df.to_csv(file_path, index=False)

def append_df_row(row, expected_columns, file_path):
    """Append a single row (as a dict) to a CSV file, using a fixed column order.
    Missing columns are filled with NaN.
    """
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=expected_columns)
    header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
    df_row.to_csv(file_path, mode="a", header=header, index=False)
