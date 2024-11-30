# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:17:24 2024

Revised and optimized script for building a more accurate and faster BPNN model.

Tuned ande Scaled Model

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.mixed_precision import Policy, set_global_policy
import joblib
import matplotlib.pyplot as plt

# Enable mixed precision training
policy = Policy('mixed_float16')
set_global_policy(policy)

# Step 1: Load data from a .pkl file
def load_data(file_path):
    print(f"Loading data from {file_path}")
    data = pd.read_pickle(file_path)
    return data

# Step 2: Preprocess data
def preprocess_data(data, handedness):
    # Filter by handedness
    data = data[data['p_throws'] == handedness]
    
    features = [
        'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'spin_axis', 
        'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in'
    ]
    target = 'pitch_type'

    data = data.dropna(subset=features + [target])  # Drop rows with missing values

    # Encode target variable
    le = LabelEncoder()
    data.loc[:, 'pitch_type_encoded'] = le.fit_transform(data[target])

    # Extract features and target
    X = data[features]
    y = data['pitch_type_encoded']

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le, scaler

# Step 3: Define the neural network
def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(2048),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1024),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(512),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.2),

        Dense(256),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])
    optimizer = AdamW(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(X, y, le, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode the target variable
    y_train = to_categorical(y_train, num_classes=len(le.classes_))
    y_test = to_categorical(y_test, num_classes=len(le.classes_))

    # Build the model
    model = build_model(input_dim=X_train.shape[1], num_classes=len(le.classes_))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f"{model_name}.keras", monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy for {model_name}: {accuracy * 100:.2f}%")

    # Plot the training history
    plot_training_history(history)

    return model, history

def plot_training_history(history):
    # Extract accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 5: Save the model and encoders
def save_model_and_encoders(model, le, scaler, model_name):
    print(f"Saving the model to {model_name}.keras")
    model.save(f"{model_name}.keras")
    joblib.dump(le, f"{model_name}_label_encoder.pkl")
    joblib.dump(scaler, f"{model_name}_scaler.pkl")
    print("Model, label encoder, and scaler saved successfully.")

# Main script
if __name__ == "__main__":
    # Step 1: Load data
    file_path = "./pitch_data/pitch_data_2024.pkl"
    data = load_data(file_path)

    # Train separate models for left-handed and right-handed pitchers
    for handedness in ['L', 'R']:
        print(f"\nProcessing data for pitchers who throw {handedness}...\n")
        X, y, le, scaler = preprocess_data(data, handedness)
        model_name = f"pitch_classifier_model_tuned_{handedness}"
        model, history = train_and_evaluate(X, y, le, model_name)
        save_model_and_encoders(model, le, scaler, model_name)
