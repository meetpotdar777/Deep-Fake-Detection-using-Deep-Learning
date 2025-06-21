# main.py - Deep Fake Detection using Deep Learning

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define image parameters for our synthetic data
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3 # RGB images
NUM_CLASSES = 2 # 0: Real, 1: Fake

# --- 1. Synthetic Data Generation ---
def generate_synthetic_image_data(num_samples=2000):
    """
    Generates synthetic 'real' and 'fake' image-like data.
    'Real' images will have a smooth gradient or simple pattern.
    'Fake' images will have more abrupt changes, noise, or specific 'artifacts'.
    """
    print(f"Generating {num_samples} synthetic image samples...")
    
    X = np.zeros((num_samples, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.int32) # 0 for real, 1 for fake

    # Generate 'real' images (first half of samples)
    for i in range(num_samples // 2):
        # Create a smooth gradient pattern
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
        for c in range(CHANNELS):
            # Gradient from top-left to bottom-right
            for r in range(IMG_HEIGHT):
                for col in range(IMG_WIDTH):
                    value = (r / IMG_HEIGHT + col / IMG_WIDTH) / 2.0
                    img[r, col, c] = value + np.random.uniform(-0.1, 0.1) # Add slight noise
        X[i] = np.clip(img, 0, 1) # Clip values to [0, 1]
        y[i] = 0 # Label as real

    # Generate 'fake' images (second half of samples)
    for i in range(num_samples // 2, num_samples):
        # Create a noisy, patchy pattern, simulating artifacts
        img = np.random.rand(IMG_HEIGHT, IMG_WIDTH, CHANNELS).astype(np.float32) * 0.5 # Base noise
        
        # Add a "fake" artifact: a brighter/darker central square or circle with sharp edges
        center_row, center_col = np.random.randint(IMG_HEIGHT // 4, IMG_HEIGHT * 3 // 4, 2)
        artifact_size = np.random.randint(IMG_HEIGHT // 8, IMG_HEIGHT // 4)
        
        for r in range(IMG_HEIGHT):
            for col in range(IMG_WIDTH):
                if (r - center_row)**2 + (col - center_col)**2 < artifact_size**2: # Circular artifact
                    img[r, col, :] += np.random.uniform(0.5, 1.0) # Make it brighter
                else:
                    img[r, col, :] += np.random.uniform(0, 0.2) # General background noise
        X[i] = np.clip(img, 0, 1) # Clip values to [0, 1]
        y[i] = 1 # Label as fake
    
    print("Synthetic data generation complete.")
    return X, y

# --- 2. Build Deep Learning Model (CNN) ---
def build_cnn_model():
    """
    Builds a simple Convolutional Neural Network (CNN) model using Keras.
    """
    print("Building CNN model...")
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Layer 3 (Optional, for deeper features)
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten layer to prepare for Dense layers
        layers.Flatten(),
        
        # Dense (Fully Connected) Layer
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization to prevent overfitting
        
        # Output Layer: 1 neuron for binary classification, sigmoid activation for probabilities
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    # Optimizer: Adam is a good general-purpose optimizer
    # Loss: binary_crossentropy for binary classification
    # Metrics: accuracy to monitor performance during training
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("CNN model built successfully.")
    model.summary() # Print model summary
    return model

# --- 3. Train the Model ---
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Trains the deep learning model.
    """
    print(f"\nTraining model for {epochs} epochs...")
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_split=0.2, # Use 20% of training data for validation during training
                        verbose=1)
    print("Model training complete.")
    return history

# --- 4. Evaluate the Model ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    """
    print("\n--- Model Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot Confusion Matrix for better visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Real', 'Predicted Fake'], 
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print("Evaluation complete. Confusion Matrix displayed.")

# --- 5. Prediction Function ---
def predict_deepfake(model, image_data):
    """
    Predicts whether a given image is 'real' or 'fake'.
    Args:
        model: The trained Keras CNN model.
        image_data (np.array): A single image array (IMG_HEIGHT, IMG_WIDTH, CHANNELS).
                               Should be preprocessed (normalized to 0-1).
    Returns:
        tuple: (prediction_label_string, probability_of_fake_float)
    """
    if image_data.shape != (IMG_HEIGHT, IMG_WIDTH, CHANNELS):
        raise ValueError(f"Input image shape must be ({IMG_HEIGHT}, {IMG_WIDTH}, {CHANNELS}). Got {image_data.shape}")
    
    # Model expects a batch of images, so add a batch dimension
    input_image = np.expand_dims(image_data, axis=0) 
    
    prediction_proba = model.predict(input_image, verbose=0)[0][0] # Get probability of being fake
    
    if prediction_proba > 0.5:
        prediction_label = "Fake"
    else:
        prediction_label = "Real"
        
    print(f"\nPrediction: The image is likely {prediction_label} (Fake Probability: {prediction_proba:.2%})")
    
    return prediction_label, prediction_proba

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Deep Fake Detection System ---")

    # 1. Generate Synthetic Data
    X, y = generate_synthetic_image_data(num_samples=2000) # Generating 2000 samples

    # Display some sample images
    plt.figure(figsize=(10, 5))
    for i in range(5): # Display 5 real images
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i])
        plt.title(f"Real")
        plt.axis('off')
    for i in range(5): # Display 5 fake images
        plt.subplot(2, 5, i + 6)
        plt.imshow(X[len(X)//2 + i]) # Start from the first fake image
        plt.title(f"Fake")
        plt.axis('off')
    plt.suptitle("Sample Synthetic Images (Top: Real, Bottom: Fake)")
    plt.show()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # 2. Build Model
    model = build_cnn_model()

    # 3. Train Model
    # Increased epochs for better learning on synthetic data
    history = train_model(model, X_train, y_train, epochs=15) 

    # 4. Evaluate Model
    evaluate_model(model, X_test, y_test)

    # 5. Test Prediction on a few specific synthetic examples
    print("\n--- Testing Predictions on Specific Examples ---")
    
    # Test with a 'real' image from the test set
    real_test_idx = np.where(y_test == 0)[0][0] # Find first real image in test set
    print(f"\nTesting with an actual 'Real' image (index {real_test_idx} in test set):")
    predict_deepfake(model, X_test[real_test_idx])
    
    # Test with a 'fake' image from the test set
    fake_test_idx = np.where(y_test == 1)[0][0] # Find first fake image in test set
    print(f"\nTesting with an actual 'Fake' image (index {fake_test_idx} in test set):")
    predict_deepfake(model, X_test[fake_test_idx])

    print("\n--- Deep Fake Detection System operations completed. ---")

# Note: This code is designed to run in an environment with TensorFlow and Keras installed.