import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load Dataset
# Sesuaikan path jika file csv Anda berada di folder berbeda
df = pd.read_csv('dataset/hand_landmarks_data.csv')

# 2. Pisahkan Fitur (X) dan Label (y)
# X adalah kolom x1, y1, z1 sampai x21, y21, z21
# y adalah kolom 'label'
X = df.drop('label', axis=1)
y = df['label']

# 3. Encode Label (Mengubah teks menjadi angka)
# Contoh: 'peace' jadi 1, 'fist' jadi 2, dll.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Memulai proses training...")

# 5. Training Model Random Forest
# Kita gunakan Random Forest karena akurasinya mencapai ~95% di project ini
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi Sederhana
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Training selesai! Akurasi Model: {accuracy * 100:.2f}%")

# 7. SIMPAN MODEL DAN ENCODER
# Ini adalah bagian terpenting agar bisa dipakai di app.py
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("File model.pkl dan label_encoder.pkl berhasil dibuat!")