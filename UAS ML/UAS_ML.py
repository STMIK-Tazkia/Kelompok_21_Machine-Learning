import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================
# 1. PERSIAPAN DATA (PREPROCESSING)
# ==========================================
print("--- 1. Memuat & Menyiapkan Data ---")

# Load Data
df_train = pd.read_csv('train.csv')

# Pisahkan Fitur (X) dan Target (y)
X = df_train.drop(['critical_temp'], axis=1)
y = df_train['critical_temp']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [PENTING UNTUK DEEP LEARNING] Scaling Data
# Neural Network sangat sensitif terhadap angka besar/kecil.
# Kita harus mengubah semua angka menjadi skala standar (sekitar -1 sampai 1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data siap! Melakukan Training Neural Network...")

# ==========================================
# 2. MEMBANGUN ARSITEKTUR (MENJAWAB RM NO. 2)
# ==========================================
# Kita membuat 'Deep' Neural Network dengan 3 Hidden Layers.
# Model ini lebih kompleks daripada sekadar regresi biasa.

model = Sequential([
    # Hidden Layer 1: 128 Neuron, Aktivasi ReLU
    # Input shape 81 sesuai jumlah fitur kimia kamu
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    
    # Hidden Layer 2: 64 Neuron
    Dense(64, activation='relu'),
    
    # Hidden Layer 3: 32 Neuron (Makin mengerucut)
    Dense(32, activation='relu'),
    
    # [MENJAWAB RM NO. 3] Dropout Layer
    # Mematikan 20% neuron secara acak agar tidak 'Overfitting' (terlalu menghapal)
    Dropout(0.2),

    # Output Layer: 1 Neuron (Karena kita prediksi 1 angka suhu)
    Dense(1, activation='linear') 
])

# Kompilasi Model (Menentukan cara belajar)
model.compile(optimizer='adam', loss='mean_squared_error')

# ==========================================
# 3. TRAINING DENGAN SAFETY BRAKE (MENJAWAB RM NO. 3)
# ==========================================
# EarlyStopping: Berhenti otomatis jika model tidak makin pintar (biar hemat waktu)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Mulai Training
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,          # Maksimal 100 kali belajar
    batch_size=32,       # Belajar per 32 data
    callbacks=[early_stop], 
    verbose=1
)

# ==========================================
# 4. EVALUASI HASIL (MENJAWAB RM NO. 1)
# ==========================================
print("\n--- Evaluasi Model Deep Learning ---")

# Prediksi data test
y_pred = model.predict(X_test_scaled).flatten()

# Hitung Metrik
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score (Akurasi): {r2:.4f}")
print(f"RMSE (Error): {rmse:.4f}")
print(f"MAE (Rata-rata Meleset): {mae:.4f} Kelvin")

# ==========================================
# 5. VISUALISASI (BUKTI UNTUK LAPORAN)
# ==========================================

# Grafik 1: Loss Curve (Bukti Kestabilan - RM No. 3)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (Error Belajar)')
plt.plot(history.history['val_loss'], label='Validation Loss (Error Ujian)')
plt.title('Grafik Penurunan Error (Loss Curve)')
plt.xlabel('Epoch (Putaran Belajar)')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Grafik 2: Prediksi vs Asli (Bukti Kinerja - RM No. 1)
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Garis diagonal sempurna
plt.title(f'Prediksi vs Aktual (R2: {r2:.2f})')
plt.xlabel('Suhu Asli (Target)')
plt.ylabel('Suhu Prediksi Model')
plt.grid(True)

plt.tight_layout()
plt.show()