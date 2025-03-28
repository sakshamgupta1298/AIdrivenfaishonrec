import cv2
import numpy as np
from gender_detection.model import GenderDetectionModel
from fashion_recommender.recommender import FashionRecommender
import os
from pathlib import Path

class FashionAI:
    def __init__(self):
        self.gender_model = GenderDetectionModel()
        self.fashion_recommender = FashionRecommender()
        
        # Load pre-trained model if it exists
        model_path = 'data/gender_model.h5'
        if os.path.exists(model_path):
            self.gender_model.load_model(model_path)
            
        # Load face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_face(self, image):
        """Detect face in the image using OpenCV with adjusted parameters."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Adjust image contrast
        gray = cv2.equalizeHist(gray)
        
        # Try different scale factors and min neighbors
        scale_factors = [1.05, 1.1, 1.15, 1.2]
        min_neighbors_values = [3, 4, 5]
        
        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_values:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # Get the largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    (x, y, w, h) = largest_face
                    
                    # Add padding around the face
                    padding = int(0.4 * w)  # 40% padding to include more hair and facial features
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    
                    face = image[y:y+h, x:x+w]
                    return face, (x, y, w, h)
        
        raise ValueError("No face detected in the image. Please try a different image with a clearer face.")
    
    def process_image(self, image_path):
        """Process an image to detect gender and recommend outfits."""
        # Convert path to proper format
        image_path = str(Path(image_path).resolve())
        
        # Read and preprocess image
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}. Please check if the file exists and is a valid image.")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and extract face
        face, face_coords = self.detect_face(image)
        
        # Resize face for model input
        face_resized = cv2.resize(face, (64, 64))
        
        # Detect gender
        is_female = self.gender_model.predict(face_resized)
        gender = 'female' if is_female else 'male'
        
        return gender, is_female, face, face_coords
    
    def get_recommendations(self, gender, occasion):
        """Get fashion recommendations based on gender and occasion."""
        return self.fashion_recommender.get_recommendations(gender, occasion)
    
    def display_results(self, image_path, gender, recommendations, face, face_coords):
        """Display the results with the image and recommendations."""
        # Convert path to proper format and read image
        image_path = str(Path(image_path).resolve())
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Create a window with a specific size
        cv2.namedWindow('Fashion AI Results', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fashion AI Results', 800, 600)
        
        # Create a copy for drawing
        display_image = image.copy()
        
        # Draw face rectangle
        x, y, w, h = face_coords
        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add gender text with better visibility
        text_color = (0, 255, 0)  # Green color
        text_bg_color = (0, 0, 0)  # Black background
        
        gender_text = f"Detected: {gender}"
        (text_width, text_height), _ = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(display_image, (5, 5), (text_width + 15, 40), text_bg_color, -1)
        cv2.putText(display_image, gender_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Display recommendations with better formatting
        y_pos = 70
        for i, outfit in enumerate(recommendations, 1):
            outfit_text = f"Outfit {i}: {self.fashion_recommender.get_outfit_details(outfit)}"
            # Add background rectangle for better text visibility
            (text_width, text_height), _ = cv2.getTextSize(outfit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_image, (5, y_pos - text_height - 5), (text_width + 15, y_pos + 5), text_bg_color, -1)
            cv2.putText(display_image, outfit_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
        
        # Show the image
        cv2.imshow('Fashion AI Results', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Initialize the Fashion AI system
    fashion_ai = FashionAI()
    
    while True:
        try:
            # Get image path
            print("\nPlease enter the path to your image.")
            print("You can either:")
            print("1. Enter the full path (e.g., C:\\Users\\Name\\Pictures\\photo.jpg)")
            print("2. Enter just the filename if it's in the current directory (e.g., photo.jpg)")
            image_path = input("\nEnter the path to your image: ").strip('"').strip("'")
            
            if not os.path.exists(image_path):
                print(f"Error: File not found at {image_path}")
                continue
                
            # Get occasion
            print("\nAvailable occasions: casual, formal, party")
            occasion = input("Enter the occasion: ").lower().strip()
            
            if occasion not in ['casual', 'formal', 'party']:
                print("Error: Invalid occasion. Please choose from: casual, formal, party")
                continue
            
            # Process the image
            gender, is_female, face, face_coords = fashion_ai.process_image(image_path)
            print(f"\nDetected gender: {gender}")
            
            # Get recommendations
            recommendations = fashion_ai.get_recommendations(gender, occasion)
            
            # Display results
            fashion_ai.display_results(image_path, gender, recommendations, face, face_coords)
            
            # Ask if user wants to try another image
            retry = input("\nWould you like to try another image? (yes/no): ").lower().strip()
            if retry != 'yes':
                break
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            retry = input("Would you like to try again? (yes/no): ").lower().strip()
            if retry != 'yes':
                break

if __name__ == "__main__":
    main() 