import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-10 dataset
print("📦 Loading CIFAR-10 dataset...")
(X, y), (_, _) = tf.keras.datasets.cifar10.load_data()
y = y.flatten()

# Filter only classes 3 (cat) and 5 (dog)
print("🔍 Filtering only class 3 (cat) and class 5 (dog)...")
mask = (y == 3) | (y == 5)
X = X[mask].astype('float32') / 255.0  # Normalize
y = y[mask]

# Re-label: cat → 0, dog → 1
y = tf.keras.utils.to_categorical((y == 5).astype(int), 2)

# Split into training and validation sets
print("✂️ Splitting dataset into train and validation...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build the model
print("🛠️ Creating and compiling model...")
model = create_model()

# Use EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
print("🚀 Starting training...")
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

# Print final accuracy
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"\n✅ Final Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Final Validation Accuracy: {val_acc * 100:.2f}%")

# Save the model
MODEL_PATH = "insulator_model.h5"
print(f"💾 Saving model to: {MODEL_PATH}")
model.save(MODEL_PATH)

# Confirm the file exists
if os.path.exists(MODEL_PATH):
    print(f"✅ Model saved successfully at {os.path.abspath(MODEL_PATH)}")
else:
    print("❌ Error: Model file was not saved!")
