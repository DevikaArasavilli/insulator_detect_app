import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Set seed for reproducibility
tf.random.set_seed(42)

# Paths
DATA_DIR = 'dataset'
MODEL_PATH = 'insulator_model.h5'

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 15

# Load and preprocess data
print("📦 Loading images from dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Show label mapping
print("✅ Class indices:", train_gen.class_indices)
print("📊 Normal:", len(os.listdir(os.path.join(DATA_DIR, 'normal'))))
print("📊 Faulty:", len(os.listdir(os.path.join(DATA_DIR, 'faulty'))))

# Compute class weights
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print("⚖ Class Weights:", class_weight_dict)

# Build model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])

    # ❌ Do not use weight_decay here
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train
print("🧠 Training model...")
model = create_model()
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# Print formula & accuracy
print("\n📐 Accuracy = (Correct / Total) × 100")
train_acc = round(history.history['accuracy'][-1] * 100, 2)
val_acc = round(history.history['val_accuracy'][-1] * 100, 2)
print(f"✅ Final Training Accuracy: {train_acc}%")
print(f"✅ Final Validation Accuracy: {val_acc}%")

# Save accuracy for display
with open("accuracy.txt", "w") as f:
    f.write(str(val_acc))

# Save model (safe format)
print(f"💾 Saving model to: {MODEL_PATH}")
model.save(MODEL_PATH)

if os.path.exists(MODEL_PATH):
    print("✅ Model saved successfully.")
else:
    print("❌ Error saving model.")
