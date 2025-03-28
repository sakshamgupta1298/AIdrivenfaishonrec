import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from gender_detection.model import GenderDetectionModel
from tqdm import tqdm
import gc

def detect_face(image, face_cascade):
    """Detect and extract face from image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
        
    # Get the first face detected
    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    return face

def load_dataset(data_dir, batch_size=32):
    """Load and preprocess the dataset with batching."""
    images = []
    labels = []
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Process male images
    male_dir = os.path.join(data_dir, 'male')
    male_files = [f for f in os.listdir(male_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(male_files)} male images")
    
    print("\nProcessing male images...")
    for img_name in tqdm(male_files):
        img_path = os.path.join(male_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = detect_face(img, face_cascade)
                if face is not None:
                    face = cv2.resize(face, (64, 64))
                    images.append(face)
                    labels.append(0)  # 0 for male
        except Exception as e:
            print(f"Error processing male image {img_name}: {str(e)}")
    
    # Process female images
    female_dir = os.path.join(data_dir, 'female')
    female_files = [f for f in os.listdir(female_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(female_files)} female images")
    
    print("\nProcessing female images...")
    for img_name in tqdm(female_files):
        img_path = os.path.join(female_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = detect_face(img, face_cascade)
                if face is not None:
                    face = cv2.resize(face, (64, 64))
                    images.append(face)
                    labels.append(1)  # 1 for female
        except Exception as e:
            print(f"Error processing female image {img_name}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\nDataset summary:")
    print(f"Total images processed: {len(X)}")
    print(f"Male images: {np.sum(y == 0)}")
    print(f"Female images: {np.sum(y == 1)}")
    
    return X, y

def main():
    # Initialize the model
    model = GenderDetectionModel()
    
    # Load dataset
    print("Loading dataset...")
    data_dir = "data/dataset"  # Update this path to your dataset directory
    
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
    
    X, y = load_dataset(data_dir, batch_size=32)
    
    if len(X) == 0:
        print("Error: No valid faces detected in the dataset!")
        return
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the model
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Save the model
    model_path = 'data/gender_model.h5'
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Print final accuracy
    final_accuracy = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_accuracy:.2%}")

if __name__ == "__main__":
    main() 