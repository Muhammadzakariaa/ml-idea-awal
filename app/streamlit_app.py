import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Config Halaman
st.set_page_config(
    page_title="AI Smart Home & Gesture Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan premium dan dinamis (Sleek Dark Mode & Glassmorphism)
st.markdown("""
<style>
    /* Styling Dasar */
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header & Title */
    h1, h2, h3 {
        color: #00ffcc !important;
        font-weight: 700 !important;
    }
    .main-title {
        text-align: center;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(45deg, #00ffcc, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Container Box / Cards */
    .card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Smart Home Status Boxes */
    .room-container {
        display: flex;
        gap: 20px;
        justify-content: space-between;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .room-card {
        flex: 1;
        min-width: 200px;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid #334155;
        background: #1e293b;
        color: #ffffff;
    }
    
    /* Active States */
    .active-dapur {
        background: #ea580c !important;
        color: #ffffff !important;
        border-color: #ffedd5 !important;
        box-shadow: 0 0 25px rgba(234, 88, 12, 0.6);
    }
    .active-tamu {
        background: #eab308 !important;
        color: #000000 !important;
        border-color: #fef9c3 !important;
        box-shadow: 0 0 25px rgba(234, 179, 8, 0.6);
    }
    .active-kamar {
        background: #06b6d4 !important;
        color: #ffffff !important;
        border-color: #ecfeff !important;
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODEL & RESOURCES -----------------
@st.cache_resource
def load_models():
    # Model Paths
    svm_path = 'models/best_model_svm.pkl'
    rf_path = 'models/model.pkl'
    le_path = 'models/label_encoder.pkl'
    
    # Fallback to alternative folders if running from different directories
    if not os.path.exists(svm_path):
        svm_path = '../models/best_model_svm.pkl'
        rf_path = '../models/model.pkl'
        le_path = '../models/label_encoder.pkl'
        
    model = None
    model_name = "None"
    
    # Pilih model terbaik (SVM) sebagai prioritas utama
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            model = pickle.load(f)
        model_name = "Support Vector Machine (SVM) - Best Model"
    elif os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            model = pickle.load(f)
        model_name = "Random Forest - Baseline Model"
        
    le = None
    if os.path.exists(le_path):
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
            
    return model, le, model_name

model, le, active_model_name = load_models()

@st.cache_data
def load_dataset():
    path = 'data/processed/hand_landmarks_data_clean.csv'
    if not os.path.exists(path):
        path = '../data/processed/hand_landmarks_data_clean.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_clean = load_dataset()

# ----------------- TITLE -----------------
st.markdown("<h1 class='main-title'>🤖 AI Smart Home & Hand Gesture Analytics</h1>", unsafe_allow_html=True)
st.sidebar.title("Navigasi")
tab_selection = st.sidebar.radio("Pilih Menu:", [
    "📈 Dashboard EDA",
    "📷 Demo Prediksi Saklar",
    "📊 Evaluasi Model",
    "💡 Interpretasi Model (SHAP)",
    "📖 Dokumentasi Proyek"
])

# ----------------- TAB 1: DASHBOARD EDA -----------------
if tab_selection == "📈 Dashboard EDA":
    st.markdown("## 📈 Dashboard Analisis Data Eksploratif (EDA)")
    st.markdown("Eksplorasi dataset koordinat landmark tangan hasil ekstraksi MediaPipe Hands.")
    
    if df_clean is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sampel Data", f"{df_clean.shape[0]:,}")
        with col2:
            st.metric("Jumlah Fitur Koordinat", f"{df_clean.shape[1] - 1} (21 Landmark x 3)")
        with col3:
            st.metric("Total Kelas Gestur", f"{df_clean['label'].nunique()}")
            
        st.markdown("### 📊 Distribusi Kelas Gestur di Dataset")
        class_counts = df_clean['label'].value_counts().reset_index()
        class_counts.columns = ['Gestur', 'Jumlah Sampel']
        fig_dist = px.bar(
            class_counts, 
            x='Gestur', 
            y='Jumlah Sampel', 
            color='Jumlah Sampel',
            color_continuous_scale='teal',
            title='Distribusi Jumlah Sampel per Gerakan Tangan'
        )
        fig_dist.update_layout(template='plotly_dark')
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # 3D HAND LANDMARK VISUALIZER
        st.markdown("### 🖐️ Visualisasi 3D Landmark Koordinat Tangan")
        st.markdown("Pilih gestur dari dataset untuk melihat visualisasi sendi dan tulang tangan dalam ruang 3D.")
        
        selected_gesture = st.selectbox("Pilih Kelas Gestur:", df_clean['label'].unique())
        sample_row = df_clean[df_clean['label'] == selected_gesture].iloc[0]
        
        # Extract X, Y, Z
        xs = [sample_row[f'x{i}'] for i in range(1, 22)]
        ys = [sample_row[f'y{i}'] for i in range(1, 22)]
        zs = [sample_row[f'z{i}'] for i in range(1, 22)]
        
        # MediaPipe Connection lines
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Ibu jari
            (0, 5), (5, 6), (6, 7), (7, 8),      # Jari Telunjuk
            (5, 9), (9, 10), (10, 11), (11, 12),  # Jari Tengah
            (9, 13), (13, 14), (14, 15), (15, 16),# Jari Manis
            (13, 17), (17, 18), (18, 19), (19, 20),# Jari Kelingking
            (0, 17)                              # Telapak bagian bawah
        ]
        
        fig_3d = go.Figure()
        
        # Add joints (scatter points)
        fig_3d.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers+text',
            marker=dict(size=6, color=ys, colorscale='Viridis', opacity=0.8),
            text=[f"Pt {i}" for i in range(1, 22)],
            name='Sendi Jari'
        ))
        
        # Add bones (lines)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            fig_3d.add_trace(go.Scatter3d(
                x=[xs[start_idx], xs[end_idx]],
                y=[ys[start_idx], ys[end_idx]],
                z=[zs[start_idx], zs[end_idx]],
                mode='lines',
                line=dict(color='cyan', width=3),
                showlegend=False
            ))
            
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Sumbu X (Lebar)',
                yaxis_title='Sumbu Y (Tinggi)',
                zaxis_title='Sumbu Z (Kedalaman)',
                xaxis=dict(autorange="reversed"), # Reverse X for mirror representation
                yaxis=dict(autorange="reversed")  # Reverse Y to match camera coordinates direction
            ),
            title=f"Struktur Tulang Tangan 3D - Gestur: {selected_gesture.upper()}",
            template='plotly_dark',
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.error("Gagal memuat dataset. Pastikan file hand_landmarks_data_clean.csv berada di direktori data/processed/.")

# ----------------- TAB 2: DEMO PREDIKSI -----------------
elif tab_selection == "📷 Demo Prediksi Saklar":
    st.markdown("## 📷 Demo Prediksi & Saklar Smart Home Real-Time")
    st.info(f"Model Aktif saat ini: **{active_model_name}**")
    
    if model is None or le is None:
        st.error("Model atau Label Encoder gagal dimuat. Pastikan file pkl ada di folder models/.")
    else:
        # Layout kolom
        col_cam, col_house = st.columns([1.2, 1])
        
        # Inisialisasi status lampu default (Semua Mati)
        status_dapur = "Mati"
        status_tamu = "Mati"
        status_kamar = "Mati"
        
        with col_cam:
            st.markdown("### 📸 Kamera Webcam Input")
            camera_img = st.camera_input("Arahkan tangan Anda ke kamera dan ambil foto:")
            
            if camera_img is not None:
                # Membaca gambar dari widget Streamlit
                img_pil = Image.open(camera_img)
                img_np = np.array(img_pil)
                
                # MediaPipe Hands
                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
                mp_draw = mp.solutions.drawing_utils
                
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Gambar landmark di layar
                        mp_draw.draw_landmarks(img_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Ekstraksi koordinat
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            # SANGAT PENTING: Dataset dicatat dalam pixel-scale (diasumsikan frame 640x480)
                            # Agar data inference sinkron dengan skala dataset training, kita kalikan X dengan 640 dan Y dengan 480!
                            landmarks.extend([lm.x * 640, lm.y * 480, lm.z])
                        
                        # Prediksi Gestur
                        prediction = model.predict([landmarks])
                        gesture_pred = le.inverse_transform(prediction)[0].lower().strip()
                        
                        st.success(f"🤖 AI Mendeteksi Gestur: **{gesture_pred.upper()}**")
                        
                        # Logika Saklar Lampu
                        if gesture_pred == 'peace':
                            status_dapur = "MENYALA (ORANGE)"
                        elif gesture_pred == 'four':
                            status_tamu = "MENYALA (KUNING)"
                        elif gesture_pred == 'two_up_inverted' or gesture_pred == 'twoupinverted':
                            status_kamar = "MENYALA (BIRU)"
                        elif gesture_pred in ['fist', 'stop']:
                            status_dapur = "Mati"
                            status_tamu = "Mati"
                            status_kamar = "Mati"
                else:
                    st.warning("⚠️ Tidak ada tangan yang terdeteksi di frame foto. Coba lagi dengan pencahayaan yang baik.")
                    
                # Tampilkan hasil gambar yang terdeteksi
                st.image(img_np, caption="Hasil Pemrosesan Landmark Tangan", use_container_width=True)
                
        with col_house:
            st.markdown("### 🏠 Panel Status Smart Home")
            st.markdown("Logika saklar otomatis:")
            st.markdown("- ✌️ **PEACE** -> Dapur ON | ✋ **FOUR** -> Ruang Tamu ON | 🤘 **TWO_UP_INVERTED** -> Kamar ON")
            st.markdown("- ✊ **FIST / STOP** -> Semua Lampu OFF")
            
            # CSS Class assignment berdasarkan status
            class_dapur = "active-dapur" if "MENYALA" in status_dapur else ""
            class_tamu = "active-tamu" if "MENYALA" in status_tamu else ""
            class_kamar = "active-kamar" if "MENYALA" in status_kamar else ""
            
            st.markdown(f"""
            <div class="room-container">
                <div class="room-card {class_dapur}">
                    <h3>KITCHEN (DAPUR)</h3>
                    <p style="font-size: 1.3em; font-weight: bold;">Lampu: {status_dapur}</p>
                </div>
                <div class="room-card {class_tamu}">
                    <h3>LIVING ROOM (RUANG TAMU)</h3>
                    <p style="font-size: 1.3em; font-weight: bold;">Lampu: {status_tamu}</p>
                </div>
                <div class="room-card {class_kamar}">
                    <h3>BEDROOM (KAMAR TIDUR)</h3>
                    <p style="font-size: 1.3em; font-weight: bold;">Lampu: {status_kamar}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tampilkan tombol simulasi alternatif jika kamera tidak diizinkan
            st.markdown("---")
            st.markdown("#### ⚙️ Simulasi Saklar Manual (Tanpa Kamera)")
            sim_gesture = st.selectbox("Pilih Gestur Simulasi:", ["PILIH GESTUR", "PEACE", "FOUR", "TWO_UP_INVERTED", "FIST", "STOP"])
            if sim_gesture != "PILIH GESTUR":
                st.markdown("### 🏠 Status Hasil Simulasi Manual:")
                s_d = "Mati"
                s_t = "Mati"
                s_k = "Mati"
                
                if sim_gesture == "PEACE":
                    s_d = "MENYALA (ORANGE)"
                elif sim_gesture == "FOUR":
                    s_t = "MENYALA (KUNING)"
                elif sim_gesture == "TWO_UP_INVERTED":
                    s_k = "MENYALA (BIRU)"
                
                c_d = "active-dapur" if "MENYALA" in s_d else ""
                c_t = "active-tamu" if "MENYALA" in s_t else ""
                c_k = "active-kamar" if "MENYALA" in s_k else ""
                
                st.markdown(f"""
                <div class="room-container">
                    <div class="room-card {c_d}">
                        <h3>KITCHEN (DAPUR)</h3>
                        <p style="font-size: 1.3em; font-weight: bold;">Lampu: {s_d}</p>
                    </div>
                    <div class="room-card {c_t}">
                        <h3>LIVING ROOM (RUANG TAMU)</h3>
                        <p style="font-size: 1.3em; font-weight: bold;">Lampu: {s_t}</p>
                    </div>
                    <div class="room-card {c_k}">
                        <h3>BEDROOM (KAMAR TIDUR)</h3>
                        <p style="font-size: 1.3em; font-weight: bold;">Lampu: {s_k}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 3: EVALUASI MODEL -----------------
elif tab_selection == "📊 Evaluasi Model":
    st.markdown("## 📊 Evaluasi Performa Model Klasifikasi")
    st.markdown("Analisis perbandingan akurasi, waktu latih, dan tingkat kesalahan deteksi.")
    
    # Tabel Perbandingan Performa
    st.markdown("### ⚔️ Perbandingan Model (SVM vs Random Forest)")
    performa_data = pd.DataFrame({
        "Model Klasifikasi": ["Random Forest (Tuned)", "Support Vector Machine (Linear SVM)"],
        "Akurasi Pengujian": ["82.03%", "93.55%"],
        "Akurasi Pelatihan": ["~98.50% (Overfitting)", "95.12%"],
        "ROC-AUC (Micro)": ["0.9782", "0.9945"],
        "Waktu Training (20k data)": ["~ 5 - 10 Menit", "~ 32 Menit"],
        "Kecepatan Inferensi per Frame": ["~ 15 ms", "< 2 ms (Sangat Ringan)"]
    })
    st.table(performa_data)
    
    st.markdown("""
    > [!NOTE]
    > **Justifikasi Model Terbaik:** Meskipun proses pelatihan SVM Linear memakan waktu lebih lama (32 menit) secara lokal, model ini dipilih sebagai model terbaik untuk deployment. Alasannya adalah akurasi testingnya jauh lebih unggul (**93.55%** vs **82.03%**), ukuran filenya lebih kecil, dan kecepatan prediksi (*inference time*) di Streamlit sangat instan (dalam milidetik).
    """)
    
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("### 🗺️ Heatmap Confusion Matrix (SVM)")
        img_cm_path = 'app/assets/images/bukti_SVM.png'
        if not os.path.exists(img_cm_path):
            img_cm_path = '../app/assets/images/bukti_SVM.png'
        if os.path.exists(img_cm_path):
            st.image(img_cm_path, use_container_width=True)
        else:
            st.warning("Grafik Confusion Matrix tidak ditemukan. Jalankan script train_and_save_all.py untuk membuatnya.")
            
    with col_roc:
        st.markdown("### 📈 Kurva ROC (Receiver Operating Characteristic)")
        img_roc_path = 'app/assets/images/bukti_roc_curve.png'
        if not os.path.exists(img_roc_path):
            img_roc_path = '../app/assets/images/bukti_roc_curve.png'
        if os.path.exists(img_roc_path):
            st.image(img_roc_path, use_container_width=True)
        else:
            st.warning("Grafik Kurva ROC tidak ditemukan. Jalankan script train_and_save_all.py untuk membuatnya.")

# ----------------- TAB 4: SHAP INTERPRETATION -----------------
elif tab_selection == "💡 Interpretasi Model (SHAP)":
    st.markdown("## 💡 Interpretasi Model & Explainable AI (XAI)")
    st.markdown("Membongkar keputusan model menggunakan metode **SHAP (SHapley Additive exPlanations)** untuk melihat fitur koordinat mana yang paling berpengaruh.")
    
    st.markdown("""
    Dalam penelitian sains data, penting untuk memberikan interpretabilitas pada model AI (*tidak sekadar menjadi kotak hitam*). 
    Metode SHAP membantu kita memahami kontribusi spasial dari 21 landmark koordinat tangan MediaPipe terhadap prediksi kelas tertentu.
    """)
    
    col_plot, col_explain = st.columns([1.2, 1])
    
    with col_plot:
        st.markdown("### 📊 Plot Kepentingan Fitur SHAP (SHAP Summary Plot)")
        img_shap_path = 'app/assets/images/shap_summary.png'
        if not os.path.exists(img_shap_path):
            img_shap_path = '../app/assets/images/shap_summary.png'
        if os.path.exists(img_shap_path):
            st.image(img_shap_path, use_container_width=True)
        else:
            st.warning("Visualisasi SHAP tidak ditemukan. Jalankan script train_and_save_all.py untuk membuatnya.")
            
    with col_explain:
        st.markdown("### 🔍 Analisis Logika Fitur Spasial")
        st.markdown("""
        Berdasarkan visualisasi SHAP untuk deteksi gestur di samping:
        
        1. **Dominasi Koordinat Sumbu Y:** Koordinat sumbu Y (seperti `y8`, `y12`, `y16`) memiliki pengaruh terbesar karena gerakan jari naik dan turun (menekuk jari) merupakan pembeda utama antara gestur tangan terbuka, mengepal, maupun membentuk simbol huruf.
        2. **Peran Jari Telunjuk (`y8`) & Tengah (`y12`):** Untuk gestur **PEACE** (✌️), koordinat Y dari ujung jari telunjuk (titik 8) dan jari tengah (titik 12) memiliki kontribusi positif yang sangat tinggi, menandakan kedua jari ini harus berada dalam posisi tegak lurus ke atas.
        3. **Peran Wrist (Pergelangan Tangan - `x1, y1`):** Koordinat pergelangan tangan bertindak sebagai jangkar geometris. Seluruh koordinat lainnya dievaluasi secara spasial relatif terhadap posisi pergelangan tangan ini.
        4. **Efisiensi Geometris:** Dengan membuktikan hanya 5-10 titik sendi jari teratas yang sangat krusial, kita bisa mengoptimalkan model di masa depan dengan hanya melatih fitur penting tersebut untuk memangkas memori komputasi lebih lanjut.
        """)

# ----------------- TAB 5: DOKUMENTASI PROYEK -----------------
else:
    st.markdown("## 📖 Dokumentasi Proyek & Metodologi")
    
    st.markdown("""
    ### 📂 Deskripsi Alur Kerja Sistem (Methodology Pipeline)
    
    Sistem cerdas ini dikembangkan melalui tahapan terstruktur berikut:
    
    1. **Akuisisi Data:** Data landmark koordinat tangan (21 titik dengan sumbu X, Y, Z) dikumpulkan secara lokal dengan merekam webcam video lalu mengekstraksinya menggunakan SDK MediaPipe.
    2. **Pra-pemrosesan Data (Preprocessing):**
        - Data dibersihkan dari anomali/missing values.
        - Label teks dienkode menjadi angka representatif menggunakan `LabelEncoder`.
        - Dataset dibagi dengan rasio 80% untuk pelatihan (*train*) dan 20% untuk pengujian (*test*).
    3. **Pelatihan & Perbandingan Model:**
        - Melatih model **Random Forest Classifier** dan melakuan penyetelan hyperparameter (*Tuning*).
        - Melatih model **Support Vector Machine (SVM)** berbasis Kernel Linear.
    4. **Interpretasi & Penjelasan (XAI):**
        - Menerapkan metode **SHAP** untuk memvisualisasikan pengaruh geometris dari koordinat jari telunjuk, tengah, dan pergelangan tangan terhadap keputusan klasifikasi AI.
    5. **Penyebaran (Deployment):**
        - Mengekspor model SVM terbaik menjadi file biner `.pkl`.
        - Membuat visualisasi Dashboard analitis dan demo kamera saklar smart home menggunakan **Streamlit**.
        
    ---
    
    ### 🖐️ Panduan Titik Landmark Tangan MediaPipe (21 Titik)
    """)
    
    st.image("https://mediapipe.dev/images/mobile/hand_landmarks.png", caption="Panduan Indeks Sendi Tangan MediaPipe (0 s.d 20)", width=450)
    
    st.markdown("""
    Setiap titik di atas direpresentasikan oleh 3 sumbu:
    - **Sumbu X:** Posisi horizontal tangan dalam frame gambar.
    - **Sumbu Y:** Posisi vertikal tangan dalam frame gambar.
    - **Sumbu Z:** Jarak kedalaman tangan relatif terhadap pergelangan tangan (*wrist*).
    """)
