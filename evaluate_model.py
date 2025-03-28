import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from gender_detection.model import GenderDetectionModel
import cv2
import os
from tqdm import tqdm
from pathlib import Path

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
        
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (64, 64))
    
    # Normalize
    image = image / 255.0
    
    return image

def evaluate_model(model_path, test_images_dir):
    """Evaluate model performance on test images."""
    # Load the model
    model = GenderDetectionModel()
    model.load_model(model_path)
    
    # Lists to store results
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    
    # Process male images
    male_dir = os.path.join(test_images_dir, 'male')
    print("\nProcessing male test images...")
    for img_path in tqdm(list(Path(male_dir).glob('*.jpg'))):
        image = load_and_preprocess_image(img_path)
        if image is not None:
            # Get prediction
            pred = model.predict(np.expand_dims(image, axis=0))
            predicted_labels.append(1 if pred else 0)
            true_labels.append(0)  # Male is 0
    
    # Process female images
    female_dir = os.path.join(test_images_dir, 'female')
    print("\nProcessing female test images...")
    for img_path in tqdm(list(Path(female_dir).glob('*.jpg'))):
        image = load_and_preprocess_image(img_path)
        if image is not None:
            # Get prediction
            pred = model.predict(np.expand_dims(image, axis=0))
            predicted_labels.append(1 if pred else 0)
            true_labels.append(1)  # Female is 1
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, 
                                      target_names=['Male', 'Female'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Male', 'Female'],
                yticklabels=['Male', 'Female'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate accuracy for each class
    male_accuracy = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    female_accuracy = conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1])
    overall_accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    
    # Print results
    print("\nModel Evaluation Results:")
    print("-" * 50)
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Male Accuracy: {male_accuracy:.2%}")
    print(f"Female Accuracy: {female_accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(class_report)
    
    # Test on a single image
    def test_single_image(image_path):
        """Test model on a single image and show confidence."""
        image = load_and_preprocess_image(image_path)
        if image is not None:
            pred = model.predict(np.expand_dims(image, axis=0))
            gender = "Female" if pred else "Male"
            
            # Display image with prediction
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.imread(image_path))
            plt.title(f'Predicted Gender: {gender}')
            plt.axis('off')
            plt.show()
            
            return gender
        return None

    return test_single_image

def main():
    # Check if model exists
    model_path = 'data/gender_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    # Check if test directory exists
    test_dir = 'data/dataset'
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return
    
    # Evaluate model
    test_single_image = evaluate_model(model_path, test_dir)
    
    # Interactive testing
    while True:
        print("\nOptions:")
        print("1. Test a specific image")
        print("2. Exit")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            image_path = input("Enter the path to the image: ")
            if os.path.exists(image_path):
                result = test_single_image(image_path)
                if result:
                    print(f"\nPredicted gender: {result}")
            else:
                print("Error: Image not found")
        elif choice == '2':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 