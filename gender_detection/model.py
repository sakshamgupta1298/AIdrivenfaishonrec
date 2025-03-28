import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class GenderDetectionModel:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block for better feature extraction
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Additional feature extraction for gender-specific traits
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')  # Changed to 2 outputs for better classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Changed loss function
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        # Add data augmentation for better training
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
        
        # Create augmented dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(1000).batch(batch_size)
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        
        # Add class weights to handle imbalanced data
        total_samples = len(y_train)
        n_females = np.sum(y_train == 1)
        n_males = total_samples - n_females
        
        class_weights = {
            0: total_samples / (2 * n_males),  # male weight
            1: total_samples / (2 * n_females)  # female weight
        }
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        return history
    
    def predict(self, image):
        """Predict gender with improved confidence handling."""
        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Ensure image is in the correct format
        if image.shape[1:] != (64, 64, 3):
            image = tf.image.resize(image, (64, 64))
        
        # Normalize pixel values
        image = image / 255.0
        
        # Get prediction probabilities for both classes
        predictions = self.model.predict(image)
        male_prob, female_prob = predictions[0]
        
        # Use female-biased threshold for better female detection
        female_threshold = 0.35  # Lower threshold favors female detection
        return female_prob > female_threshold
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = models.load_model(path) 