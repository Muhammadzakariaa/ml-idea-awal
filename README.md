# AI-Powered Smart Home Device Control using Real-Time Hand Gesture Classification based on Hand Landmarks

Repositori ini berisi Capstone Project untuk Ujian Akhir Semester (UAS) mata kuliah **Pembelajaran Mesin** Genap 2025/2026 di Universitas Dian Nuswantoro (UDINUS). 

Proyek ini mengintegrasikan pemrosesan video real-time untuk mengekstrak koordinat tangan 2D/3D (21 titik landmark = 63 fitur numerik) menggunakan **MediaPipe Hands**, lalu mengklasifikasikan gestur tangan menggunakan model **Support Vector Machine (SVM)** dan **Random Forest** untuk mengendalikan perangkat rumah pintar (saklar lampu virtual) secara instan.

---

## 📌 Latar Belakang & Masalah Penelitian

Topik penelitian ini dirumuskan untuk mengatasi keterbatasan sistem kontrol rumah pintar IoT tradisional yang biasanya bersifat pasif (remote button/CRUD) atau memerlukan hardware berspesifikasi tinggi (GPU) jika menggunakan model Deep Learning/Visi Komputer konvensional berbasis gambar mentah.

### Solusi yang Ditawarkan:
1. **Reduksi Dimensi Spasial:** Menggunakan MediaPipe Hands untuk mengekstrak 21 koordinat sendi tangan (X, Y, Z) dari kamera secara lokal, mengubah input video resolusi tinggi menjadi dataset tabular berdimensi rendah (63 fitur).
2. **Model Machine Learning Klasik yang Ringan:** Membandingkan performa Support Vector Machine (SVM) Linear dan Random Forest Classifier untuk memprediksi 18 gestur tangan secara real-time di CPU (inferensi < 10ms).
3. **Explainable AI (XAI):** Menerapkan metode **SHAP (SHapley Additive exPlanations)** untuk menganalisis kontribusi spasial dari setiap titik jari terhadap prediksi gestur.

---

## 🛠️ Struktur Repositori

Struktur repositori ini mengikuti template yang diwajibkan dalam panduan UAS:

```text
capstone-project-data-mining/
│
├── data/
│   ├── raw/                  # Dataset koordinat tangan mentah (hand_landmarks_data.csv)
│   ├── processed/            # Dataset koordinat yang telah bersih (hand_landmarks_data_clean.csv)
│   └── external/             # Data referensi eksternal (jika ada)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis & Visualisasi Landmark
│   ├── 02_modeling.ipynb     # Pelatihan Model (RF vs SVM) & Tuning Hyperparameter
│   └── 03_interpretation.ipynb # Interpretasi Model dengan SHAP
│
├── src/
│   ├── data_preprocessing.py # Script pemrosesan & normalisasi data koordinat
│   ├── train_model.py        # Script pelatihan model RF & SVM secara lokal
│   └── eksplorasi_dataset.py # Script pengujian statistik deskriptif di terminal
│
├── models/
│   ├── model.pkl             # Serialisasi model Random Forest terlatih
│   ├── best_model_svm.pkl    # Serialisasi model SVM terbaik (93.55% Akurasi)
│   └── label_encoder.pkl     # Encoder kategori nama gerakan tangan
│
├── app/
│   ├── app.py                # Aplikasi web saklar Smart Home berbasis Flask (Real-time Video Feed)
│   ├── streamlit_app.py      # Dashboard Analitis & Demo Interaktif berbasis Streamlit (UAS Soal 4)
│   ├── templates/            # Template HTML untuk aplikasi Flask
│   └── assets/               # Gambar, aset visual, dan tangkapan layar antarmuka
│
├── reports/
│   ├── Laporan_UAS.md        # Draf Laporan Teknis Lengkap format Markdown
│   └── final_report.pdf      # Laporan Akhir UAS Pembelajaran Mesin (format PDF)
│
├── requirements.txt          # Daftar dependensi pustaka Python
└── README.md                 # Dokumentasi utama proyek
```

---

## 🚀 Cara Menjalankan Proyek

### 1. Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal Python (versi 3.9 s.d 3.11 direkomendasikan) di sistem Anda.

### 2. Instalasi Dependensi
Buka terminal/powershell di root repositori ini, lalu instal dependensi dari file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi Web Smart Home (Flask)
Aplikasi ini menjalankan web server lokal untuk demonstrasi kontrol saklar lampu virtual di 3 ruangan (Dapur, Ruang Tamu, Kamar) menggunakan kamera webcam real-time:
```bash
python app/app.py
```
Akses di browser Anda melalui tautan: `http://127.0.0.1:5000`

### 4. Menjalankan Aplikasi Dashboard Analisis (Streamlit)
Aplikasi Streamlit ini memuat Dashboard EDA, Demo Prediksi Kamera statis, metrik Evaluasi Model (Confusion Matrix, ROC Curve), dan Interpretasi SHAP:
```bash
streamlit run app/streamlit_app.py
```
Akses di browser Anda melalui tautan yang muncul di terminal (biasanya `http://localhost:8501`).

---

## 📊 Ringkasan Hasil Eksperimen

- **Dataset:** 25.675 sampel data koordinat dari 18 kelas gestur tangan.
- **Rasio Split:** 80% Training (Pelatihan), 20% Testing (Pengujian).
- **Perbandingan Akurasi:**
  - **Support Vector Machine (SVM) kernel Linear:** **93.55%** (Model Terbaik untuk Deployment).
  - **Random Forest (Tuned):** **82.03%**.
- **Analisis SHAP (Interpretasi):**
  - Mengungkap bahwa koordinat jari telunjuk (`x8, y8`), jari tengah (`x12, y12`), dan ibu jari (`x4, y4`) memiliki nilai impak (SHAP value) tertinggi pada gestur seperti `peace` dan `two_up_inverted`.
