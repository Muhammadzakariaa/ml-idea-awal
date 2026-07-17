import pickle
import os

def load_pickle(file_path):
    """Fungsi utilitas untuk memuat file binary pickle."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")

def save_pickle(obj, file_path):
    """Fungsi utilitas untuk menyimpan objek ke file binary pickle."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[+] Berhasil menyimpan objek ke: {file_path}")
