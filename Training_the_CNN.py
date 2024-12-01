import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D, Concatenate, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load data
data = np.load('data.npy')  # Ensure this contains preprocessed chest X-ray images
target = np.load('target.npy')  # Ensure this contains one-hot encoded labels

# Normalize and reshape data
data = data / 255.0  # Normalize pixel values to [0, 1]

# Check input shape
input_shape = data.shape[1:]  # e.g., (50, 50, 1) for grayscale images

# Split data into training, validation, and testing sets
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1, random_state=42)
train_data, val_data, train_target, val_target = train_test_split(train_data, train_target, test_size=0.1, random_state=42)

# Define model with parallel convolutional layers
inp = Input(shape=input_shape)
parallel_kernels = [3, 5, 7]  # Parallel kernel sizes
convs = []

for kernel in parallel_kernels:
    conv = Conv2D(128, kernel, padding='same', activation='relu')(inp)
    convs.append(conv)

# Concatenate parallel layers
out = Concatenate()(convs)
conv_model = Model(inp, out)

# Add additional layers
model = Sequential()
model.add(conv_model)

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Adjust output layer for multi-class classification
num_classes = target.shape[1]  # Number of classes (e.g., 3 for COVID, Pneumonia, Normal)
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Model checkpoint to save best model
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# Train model
history = model.fit(
    train_data, train_target,
    epochs=50,
    callbacks=[checkpoint],
    validation_data=(val_data, val_target),
    batch_size=32
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'r', label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data, test_target, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
