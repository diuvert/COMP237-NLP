import pandas as pd


def try_read(path):
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Successfully loaded with encoding: {enc}")
            return df
        except Exception as e:
            print(f"Failed with encoding {enc}: {e}")
    raise ValueError("Could not read the CSV file.")


