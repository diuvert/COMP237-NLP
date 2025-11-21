import pandas as pd

def load_data(file_path):
    """
    Loads the CSV file into a pandas DataFrame.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print("\n--- Data Exploration ---")
        print("Data Shape:", df.shape)
        print("\nFirst 5 rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
