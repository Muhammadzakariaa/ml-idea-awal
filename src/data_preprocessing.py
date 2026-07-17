import pandas as pd
import numpy as np
import os

def preprocess_dataset():
    raw_path = 'data/raw/hand_landmarks_data.csv'
    processed_path = 'data/processed/hand_landmarks_data_clean.csv'
    
    # Fallback paths
    if not os.path.exists(raw_path):
        raw_path = '../data/raw/hand_landmarks_data.csv'
        processed_path = '../data/processed/hand_landmarks_data_clean.csv'
        
    print("=" * 60)
    print("         PROSES PREPROCESSING & DATA CLEANING         ")
    print("=" * 60)
    
    if not os.path.exists(raw_path):
        print(f"[-] Error: File mentah tidak ditemukan di: {raw_path}")
        return
        
    print(f"[*] Membaca data mentah dari: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"[+] Ukuran data awal: {df.shape[0]} baris, {df.shape[1]} kolom")
    
    # 1. Menghilangkan data kosong (Missing Values)
    print("[*] Memeriksa missing values...")
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[!] Ditemukan {missing_count} missing values. Melakukan dropping...")
        df = df.dropna()
    else:
        print("[+] Bersih! Tidak ditemukan missing values.")
        
    # 2. Menghilangkan data duplikat (Duplicates)
    print("[*] Memeriksa data duplikat...")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"[!] Ditemukan {duplicate_count} baris duplikat. Melakukan dropping...")
        df = df.drop_duplicates()
    else:
        print("[+] Bersih! Tidak ditemukan data duplikat.")
        
    # 3. Konsistensi Label
    print("[*] Memeriksa konsistensi penulisan label target...")
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    print(f"[+] Daftar kelas gestur unik ({df['label'].nunique()} kelas):")
    print(df['label'].value_counts())
    
    # 4. Menyimpan data bersih
    print(f"[*] Menyimpan data bersih ke: {processed_path}")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[+] Selesai! Ukuran data akhir: {df.shape[0]} baris")
    print("=" * 60)

if __name__ == '__main__':
    preprocess_dataset()
