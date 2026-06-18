import numpy as np
import pickle
import time
import os

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'label_encoder.pkl'

def load_resources():
    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
        le = pickle.load(open(ENCODER_PATH, 'rb'))
        return model, le
    except FileNotFoundError:
        print(f"[-] Error: File '{MODEL_PATH}' atau '{ENCODER_PATH}' tidak ditemukan.")
        print("[*] Pastikan file tersebut berada di folder yang sama dengan script ini.")
        return None, None

def generate_ascii_bar(value, max_value=1.0, length=20):
    """Membuat grafik batang sederhana di terminal berbasis karakter ASCII"""
    filled_length = int(length * value / max_value)
    filled_length = max(0, min(length, filled_length)) # batasi agar tidak error
    bar = '█' * filled_length + '░' * (length - filled_length)
    return bar

def analyze_coordinates():
    model, le = load_resources()
    if not model or not le:
        return

    # Ambil semua daftar gerakan yang terdaftar di label encoder
    daftar_gerakan = le.classes_
    
    print("=" * 65)
    print("      SISTEM ANALISIS PERGERAKAN KOORDINAT (X, Y) DI TERMINAL     ")
    print("=" * 65)
    print(f"[+] Model berhasil dimuat. Terdeteksi {len(daftar_gerakan)} jenis gerakan.")
    print("[*] Memulai simulasi pembacaan koordinat MediaPipe...")
    print("    Menghitung kemiripan pergerakan koordinat terhadap model...")
    print("-" * 65)
    time.sleep(2)

    try:
        # Simulasi perulangan frame (seperti saat webcam berjalan)
        for frame in range(1, 11):
            # Membersihkan terminal agar animasi terlihat interaktif di satu tempat
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 65)
            print(f" ANALISIS TIMELINE - FRAME KE: {frame} ")
            print("=" * 65)

            # 1. Menyimulasikan pergerakan koordinat tangan (21 titik landmark * 3 koordinat [x,y,z] = 63 nilai)
            # Di proyek asli, ini adalah data dari hand_landmarks Anda
            landmarks_simulasi = np.random.rand(63) 
            
            # Tambahkan sedikit fluktuasi acak pada sumbu X dan Y tertentu
            # Sumbu X (indeks genap), Sumbu Y (indeks ganjil)
            pergerakan_x = np.mean(landmarks_simulasi[0::3])
            pergerakan_y = np.mean(landmarks_simulasi[1::3])
            
            print(f"Rata-rata Posisi Tangan saat ini -> Sumbu X: {pergerakan_x:.2f} | Sumbu Y: {pergerakan_y:.2f}")
            print("-" * 65)

            # 2. Melakukan Prediksi / Pengecekan Kemiripan
            # Catatan: Jika model Anda mendukung predict_proba, kita bisa melihat persentase kemiripan tiap kelas
            if hasattr(model, "predict_proba"):
                probabilitas = model.predict_proba([landmarks_simulasi])[0]
                
                print("Tingkat Kemiripan dengan Variasi Gerakan di Dataset:")
                for idx, gesture_name in enumerate(daftar_gerakan):
                    skor_kemiripan = probabilitas[idx]
                    grafik_batang = generate_ascii_bar(skor_kemiripan)
                    print(f"  [{grafik_batang}] {gesture_name:<15} : {skor_kemiripan*100:>5.1f}% mirip")
                
                # Hasil akhir frame ini
                prediksi_idx = np.argmax(probabilitas)
                output_akhir = le.inverse_transform([prediksi_idx])[0]
                print("-" * 65)
                print(f"==> OUTPUT TERINGKAS: Koordinat X/Y cenderung mirip gerakan: [{output_akhir.upper()}]")
            
            else:
                # Jika model tidak mendukung probabilitas (misal Perceptron/SVM linear murni)
                prediksi = model.predict([landmarks_simulasi])
                output_akhir = le.inverse_transform(prediksi)[0]
                print(f"==> KESIMPULAN: Pola koordinat X dan Y menghasilkan output: {output_akhir}")
                print("    (Model Anda tidak mendukung visualisasi persentase kecocokan secara detail)")

            print("=" * 65)
            print("[Tekan Ctrl+C untuk berhenti] Menunggu frame berikutnya...")
            time.sleep(1.5) # Jeda waktu antar frame simulasi di terminal

    except KeyboardInterrupt:
        print("\n[-] Analisis dihentikan oleh pengguna.")

if __name__ == '__main__':
    analyze_coordinates()