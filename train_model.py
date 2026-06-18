import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# =====================================================================
# 1. LOAD DATASET
# =====================================================================
# Membaca file database koordinat tangan yang sudah Anda kumpulkan sebelumnya
df = pd.read_csv('dataset/hand_landmarks_data.csv')

# =====================================================================
# 2. SEPARASI FITUR DAN LABEL
# =====================================================================
# X: Berisi 63 fitur koordinat (x1, y1, z1 sampai x21, y21, z21)
# y: Berisi label target teks (misal: 'Peace', 'Fist', 'Open')
X = df.drop('label', axis=1)
y = df['label']

# =====================================================================
# 3. ENKODING LABEL TEXT KE ANGKA
# =====================================================================
# Mengubah teks menjadi kategori angka karena model ML hanya paham matematika
# Contoh: 'Fist' -> 0, 'Open' -> 1, 'Peace' -> 2
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =====================================================================
# 4. PEMBAGIAN DATA (DATA SPLITTING)
# =====================================================================
# Membagi data menjadi 80% untuk Belajar (Train) dan 20% untuk Ujian (Test)
# random_state=42 dikunci agar pembagian data konsisten setiap kali di-run
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("=" * 60)
print("       PROSES TRAINING MODEL HAND GESTURE YANG DIOPTIMASI       ")
print("=" * 60)
print("[*] Membaca dataset dan mempersiapkan fitur...")
print(f"[+] Total data training : {X_train.shape[0]} sampel")
print(f"[+] Total data testing  : {X_test.shape[0]} sampel")
print("-" * 60)
print("[*] Model sedang mempelajari pola koordinat tangan...")

# =====================================================================
# 5. DEKLARASI & PELATIHAN MODEL (RANDOM FOREST OPTIMIZED)
# =====================================================================
# Di sini kita batasi parameternya agar model "luwes" dan tidak kaku/overfit
model = RandomForestClassifier(
    n_estimators=200,       # Naikkan ke 200 pohon agar keputusan lebih matang
    max_depth=20,           # Dinaikkan agar model bisa mempelajari detail koordinat lebih spesifik
    min_samples_split=2,    # Mengembalikan sensitivitas percabangan pohon
    min_samples_leaf=1,     # Daun terakhir boleh berisi 1 sampel murni
    random_state=42,
    n_jobs=-1
)

# Perintah inti: Di baris inilah proses "belajar" matematika itu terjadi
model.fit(X_train, y_train)

# =====================================================================
# 6. EVALUASI DAN ANALISIS PERFORMA
# =====================================================================
# Menguji model yang sudah pintar dengan 20% data ujian tadi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"[+] Training selesai! Akurasi Validasi Model: {accuracy * 100:.2f}%")
print("-" * 60)
print("Laporan Performa Detail per Jenis Gerakan Tangan:")
# Menampilkan f1-score, precision, dan recall untuk tiap gerakan
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("=" * 60)

# =====================================================================
# 7. EXPORT MODEL KE FILE BINARY (.PKL)
# =====================================================================
# Menyimpan hasil pemikiran model agar bisa dipanggil secara instan oleh app.py
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("[INFO] File 'model.pkl' dan 'label_encoder.pkl' BERHASIL diperbarui!")
print("[INFO] Silakan jalankan kembali app.py Anda.")
print("=" * 60)