import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Impor helper dari utils
from utils import load_pickle

def evaluate():
    print("=" * 60)
    print("         PROSES EVALUASI MODEL MANDIRI         ")
    print("=" * 60)
    
    # 1. Tentukan path data dan model
    data_path = 'data/processed/hand_landmarks_data_clean.csv'
    model_path = 'models/best_model_svm.pkl'
    encoder_path = 'models/label_encoder.pkl'
    
    if not os.path.exists(data_path):
        data_path = '../data/processed/hand_landmarks_data_clean.csv'
        model_path = '../models/best_model_svm.pkl'
        encoder_path = '../models/label_encoder.pkl'
        
    # Jika SVM belum dilatih, coba baseline Random Forest
    if not os.path.exists(model_path):
        model_path = model_path.replace('best_model_svm.pkl', 'model.pkl')
        
    try:
        model = load_pickle(model_path)
        le = load_pickle(encoder_path)
        print(f"[+] Berhasil memuat model dari: {model_path}")
    except Exception as e:
        print(f"[-] Gagal memuat resource evaluasi: {e}")
        return
        
    # 2. Load data pengujian
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode label target
    y_encoded = le.transform(y)
    classes = le.classes_
    
    # Lakukan data split yang sama
    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 3. Prediksi
    print("[*] Melakukan prediksi pada data testing...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[+] Akurasi Model Evaluasi: {acc * 100:.2f}%")
    print("\n--- Laporan Klasifikasi Detail ---")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # 4. Buat Confusion Matrix
    print("[*] Membuat Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix (Akurasi: {acc*100:.2f}%)")
    plt.xlabel("Prediksi AI")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    
    # Simpan plot ke assets
    os.makedirs('app/assets/images', exist_ok=True)
    os.makedirs('../app/assets/images', exist_ok=True)
    try:
        plt.savefig('app/assets/images/bukti_SVM.png', dpi=150)
    except:
        plt.savefig('../app/assets/images/bukti_SVM.png', dpi=150)
    plt.close()
    print("[+] Plot Confusion Matrix berhasil disimpan!")
    print("=" * 60)

if __name__ == '__main__':
    evaluate()
