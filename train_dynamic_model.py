import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# ==========================================
# 1. CONFIGURAÇÕES DE CAMINHOS
# ==========================================
DATA_PATH = r'C:\Users\tatir\OneDrive\Documentos\Projetos Pessoais\TaekwondoProject\Taekwondo_Kick_Dataset_Creator\poses_dataset_dynamic.csv'
MODEL_SAVE_PATH = r'C:\Users\tatir\OneDrive\Documentos\Projetos Pessoais\TaekwondoProject\Taekwondo_Kick_Pose_Predict\model\kick_detection_model.h5'
SCALER_SAVE_PATH = r'C:\Users\tatir\OneDrive\Documentos\Projetos Pessoais\TaekwondoProject\Taekwondo_Kick_Pose_Predict\model\scaler.pkl'

def build_dynamic_model(input_shape):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train():
    print("Carregando dataset dinâmico...")
    df = pd.read_csv(DATA_PATH)

    # Remover colunas de identificação
    X = df.drop(columns=['timestamp', 'video_name', 'label'])
    y = df['label']

    print(f"Shape das features dinâmicas: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Construção do Modelo
    nn_model = build_dynamic_model(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    print("Iniciando treinamento do modelo dinâmico...")
    nn_model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Avaliação
    probs = nn_model.predict(X_test)
    preds = (probs > 0.5).astype(int).flatten()

    print("\n--- Relatório de Performance (Modelo Dinâmico) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    # Salvar
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    nn_model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\nNovo modelo dinâmico salvo em: {MODEL_SAVE_PATH}")
    print(f"Novo scaler salvo em: {SCALER_SAVE_PATH}")

if __name__ == "__main__":
    train()
