# -A-Deep-Learning-Odyssey-in-Rice-Type-Classification
pip install tensorflow matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Simulated dataset parameters
num_classes = 5  # e.g., Basmati, Jasmine, Arborio, Brown, White
img_height, img_width = 150, 150
num_train = 500
num_val = 100

# Generate random synthetic image data and labels
X_train = np.random.rand(num_train, img_height, img_width, 3)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, size=(num_train,)), num_classes)

X_val = np.random.rand(num_val, img_height, img_width, 3)
y_val = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, size=(num_val,)), num_classes)

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Rice Type Classification (Simulated)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
