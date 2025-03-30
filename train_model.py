import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from gender_detection.model import GenderDetectionModel
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

def detect_face(image, face_cascade):
    """Detect and extract face from image."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        (x, y, w, h) = largest_face
        
        # Add padding
        padding = int(0.3 * w)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        face = image[y:y+h, x:x+w]
        return face
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return None

def load_dataset(data_dir):
    """Load and preprocess the dataset with detailed logging."""
    images = []
    labels = []
    failed_images = {'male': [], 'female': []}
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Process male images
    male_dir = os.path.join(data_dir, 'male')
    if not os.path.exists(male_dir):
        print(f"Error: Male directory not found at {male_dir}")
        return np.array([]), np.array([])
    
    male_files = [f for f in os.listdir(male_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(male_files)} male images")
    
    print("\nProcessing male images...")
    for img_name in tqdm(male_files):
        img_path = os.path.join(male_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed_images['male'].append(f"{img_name}: Failed to read image")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = detect_face(img, face_cascade)
            if face is not None:
                face = cv2.resize(face, (64, 64))
                images.append(face)
                labels.append(0)  # 0 for male
            else:
                failed_images['male'].append(f"{img_name}: No face detected")
        except Exception as e:
            failed_images['male'].append(f"{img_name}: {str(e)}")
    
    # Process female images
    female_dir = os.path.join(data_dir, 'female')
    if not os.path.exists(female_dir):
        print(f"Error: Female directory not found at {female_dir}")
        return np.array([]), np.array([])
    
    female_files = [f for f in os.listdir(female_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(female_files)} female images")
    
    print("\nProcessing female images...")
    for img_name in tqdm(female_files):
        img_path = os.path.join(female_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed_images['female'].append(f"{img_name}: Failed to read image")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = detect_face(img, face_cascade)
            if face is not None:
                face = cv2.resize(face, (64, 64))
                images.append(face)
                labels.append(1)  # 1 for female
            else:
                failed_images['female'].append(f"{img_name}: No face detected")
        except Exception as e:
            failed_images['female'].append(f"{img_name}: {str(e)}")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Total images processed: {len(images)}")
    print(f"Male images: {labels.count(0)}")
    print(f"Female images: {labels.count(1)}")
    
    print("\nFailed Images:")
    print(f"Male failures: {len(failed_images['male'])}")
    print(f"Female failures: {len(failed_images['female'])}")
    
    if len(failed_images['male']) > 0 or len(failed_images['female']) > 0:
        print("\nDetailed failure log:")
        print("\nMale failures:")
        for failure in failed_images['male']:
            print(failure)
        print("\nFemale failures:")
        for failure in failed_images['female']:
            print(failure)
    
    return np.array(images), np.array(labels)

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Initialize the model
    model = GenderDetectionModel()
    
    # Load dataset
    print("Loading dataset...")
    data_dir = "data/dataset"
    
    # Check if dataset directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory not found at {data_dir}")
        print("Please create the following directory structure:")
        print("data/")
        print("  dataset/")
        print("    male/")
        print("      [male images]")
        print("    female/")
        print("      [female images]")
        return
    
    X, y = load_dataset(data_dir)
    
    if len(X) == 0:
        print("Error: No valid faces detected in the dataset!")
        return
    
    # Print class distribution
    y_int = y.astype(int)
    unique, counts = np.unique(y_int, return_counts=True)
    print("\nClass Distribution:")
    print(f"Male (0): {counts[0]} images ({counts[0]/len(y):.2%})")
    print(f"Female (1): {counts[1]} images ({counts[1]/len(y):.2%})")
    
    # Check for potential data issues
    if counts[0] < 100 or counts[1] < 100:
        print("\nWarning: Very small dataset size for one or both classes!")
        print("Consider adding more images for better training.")
    
    if abs(counts[0] - counts[1]) / len(y) > 0.2:
        print("\nWarning: Significant class imbalance detected!")
        print("Consider adding more images to the minority class.")
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split the dataset with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print("\nDataset Split:")
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    print("\nTraining Set Distribution:")
    print(f"Male samples: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train):.2%})")
    print(f"Female samples: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train):.2%})")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the model
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,  # Increased epochs with early stopping
        batch_size=32
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model on validation set
    print("\nValidation Set Evaluation:")
    val_metrics = model.evaluate(X_val, y_val)
    
    # Evaluate the model on test set
    print("\nTest Set Evaluation:")
    test_metrics = model.evaluate(X_test, y_test)
    
    # Save the model
    model_path = 'data/gender_model.h5'
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Final recommendations
    if test_metrics['male_accuracy'] < 0.9 or test_metrics['female_accuracy'] < 0.9:
        print("\nRecommendations for Improvement:")
        if test_metrics['male_accuracy'] < 0.9:
            print("- Add more male images with diverse characteristics")
        if test_metrics['female_accuracy'] < 0.9:
            print("- Add more female images with diverse characteristics")
        print("- Consider using data augmentation techniques")
        print("- Try adjusting the model architecture or hyperparameters")

if __name__ == "__main__":
    main() 