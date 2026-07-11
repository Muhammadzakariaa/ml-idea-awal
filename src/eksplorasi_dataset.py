import pandas as pd

df = pd.read_csv('../data/processed/hand_landmarks_data_clean.csv')

print("=== SHAPE ===")
print(df.shape)

print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

print("\n=== TIPE DATA ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DISTRIBUSI KELAS ===")
print(df.iloc[:, -1].value_counts())

print("\n=== STATISTIK DASAR ===")
print(df.describe())