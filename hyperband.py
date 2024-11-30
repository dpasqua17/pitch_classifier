# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:17:24 2024

Revised and optimized script for building a more accurate and faster BPNN model with Hyperband tuning.

Bigger than v13
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from keras_tuner import Hyperband
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

# Step 3: Define the model with hyperparameters
def build_model_with_hyperparameters(hp):
    alpha = hp.Choice('alpha', values=[0.01, 0.05, 0.1, 0.2, 0.3])
    dense_units = hp.Choice('dense_units', values=[512, 1024, 2048])
    dropout_rate = hp.Choice('dropout_rate', values=[0.2, 0.3, 0.4])

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(dense_units),
        LeakyReLU(alpha=alpha),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(dense_units // 2),
        LeakyReLU(alpha=alpha),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(dense_units // 4),
        LeakyReLU(alpha=alpha),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(num_classes, activation='softmax')
    ])
    optimizer = AdamW(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Perform hyperparameter tuning
def tune_hyperparameters(X, y, num_classes, input_dim, model_name):
    # Define tuner
    tuner = Hyperband(
        lambda hp: build_model_with_hyperparameters(hp),
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='hyperband_tuning',
        project_name=model_name
    )

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Search for the best hyperparameters
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=128,
        epochs=25,
        verbose=1
    )

    # Retrieve best hyperparameters and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    print(f"Best alpha: {best_hps.get('alpha')}")
    print(f"Best dense_units: {best_hps.get('dense_units')}")
    print(f"Best dropout_rate: {best_hps.get('dropout_rate')}")

    return best_model

# Step 5: Plot training history
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

# Step 6: Save the model and encoders
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

    for handedness in ['L', 'R']:
        print(f"\nProcessing data for pitchers who throw {handedness}...\n")
        X, y, le, scaler = preprocess_data(data, handedness)

        num_classes = len(le.classes_)
        input_dim = X.shape[1]
        model_name = f"pitch_classifier_hyperband_{handedness}"

        # Perform hyperparameter tuning
        best_model = tune_hyperparameters(X, y, num_classes, input_dim, model_name)

        # Save the best model and encoders
        save_model_and_encoders(best_model, le, scaler, model_name)
