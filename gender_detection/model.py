import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

class GenderDetectionModel:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_augmentation(self):
        return tf.keras.Sequential([
            # Basic preprocessing
            layers.Rescaling(1./255),
            
            # Essential geometric transformations
            layers.RandomFlip("horizontal"),
            
            # Standard augmentation layers
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
            
            # Add slight noise for robustness
            layers.GaussianNoise(0.01)
        ])
    
    def _focal_loss(self, gamma=2.0, alpha=0.25):
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate focal loss
            cross_entropy = -y_true * tf.math.log(y_pred)
            focal_weight = tf.pow(1 - y_pred, gamma) * y_true
            focal_loss = focal_weight * cross_entropy
            
            # Apply alpha balancing
            alpha_weight = alpha * y_true + (1 - alpha) * (1 - y_true)
            balanced_focal_loss = alpha_weight * focal_loss
            
            return tf.reduce_mean(tf.reduce_sum(balanced_focal_loss, axis=-1))
        return focal_loss_fn
    
    def _se_block(self, input_tensor, ratio=16):
        """Squeeze-and-Excitation block."""
        channels = input_tensor.shape[-1]
        
        # Squeeze operation
        x = layers.GlobalAveragePooling2D()(input_tensor)
        
        # Excitation operation
        x = layers.Dense(channels // ratio, activation='relu')(x)
        x = layers.Dense(channels, activation='sigmoid')(x)
        
        # Reshape to broadcasting shape
        x = layers.Reshape((1, 1, channels))(x)
        
        # Scale the input
        return layers.multiply([input_tensor, x])
    
    def _build_model(self):
        # Use EfficientNetV2S as base model
        base_model = applications.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=(64, 64, 3)
        )
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Create model
        inputs = layers.Input(shape=(64, 64, 3))
        
        # Preprocessing
        x = layers.GaussianNoise(0.01)(inputs)
        
        # Pass through base model
        x = base_model(x)
        
        # Apply SE attention to the output
        x = self._se_block(x)
        
        # Global pooling with concatenation
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers with Layer Normalization and optimized dropout
        x = layers.Dense(1024)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(512)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Final classification layers
        x = layers.Dense(256)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output with balanced initialization
        outputs = layers.Dense(2, activation='softmax', 
                             bias_initializer='zeros',
                             kernel_initializer='he_uniform')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Cosine annealing learning rate schedule
        initial_learning_rate = 0.001
        first_decay_steps = 1000
        
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps,
            t_mul=2.0,  # Multiply decay steps by this factor after each restart
            m_mul=0.9,  # Multiply learning rate by this factor at each restart
            alpha=1e-5  # Minimum learning rate
        )
        
        # Compile with focal loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=self._focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(class_id=1, name='female_precision'),
                tf.keras.metrics.Recall(class_id=1, name='female_recall'),
                tf.keras.metrics.Precision(class_id=0, name='male_precision'),
                tf.keras.metrics.Recall(class_id=0, name='male_recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        print("\nTraining Data Statistics:")
        print(f"Total training samples: {len(X_train)}")
        print(f"Male samples in training: {np.sum(y_train == 0)}")
        print(f"Female samples in training: {np.sum(y_train == 1)}")
        
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
        y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
        
        # Create augmentation pipeline
        augmentation = self._build_augmentation()
        
        # Create training dataset with augmentation
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
        train_ds = train_ds.shuffle(10000)
        
        # Apply augmentation with multiple versions
        def augment_multiple(image, label):
            # Original image with basic preprocessing
            orig_image = layers.Rescaling(1./255)(image)
            
            # Two augmented versions
            aug_image1 = augmentation(image, training=True)
            aug_image2 = augmentation(image, training=True)
            
            # Stack all versions
            images = tf.stack([orig_image, aug_image1, aug_image2])
            labels = tf.stack([label, label, label])
            
            return images, labels
        
        train_ds = train_ds.map(
            augment_multiple,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.unbatch()
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset with only basic preprocessing
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
        val_ds = val_ds.map(
            lambda x, y: (layers.Rescaling(1./255)(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        
        # Add callbacks with improved monitoring
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=20,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.weights.h5',
                monitor='val_auc',
                save_weights_only=True,
                mode='max'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                update_freq='epoch',
                profile_batch=0
            )
        ]
        
        # Initial training
        print("\nPhase 1: Initial training...")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best weights
        self.model.load_weights('best_model.weights.h5')
        
        # Fine-tuning phase 1: Train last 50 layers
        print("\nPhase 2: Fine-tuning last 50 layers...")
        base_model = self.model.layers[2]
        for layer in base_model.layers[-50:]:
            layer.trainable = True
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=self._focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(class_id=1, name='female_precision'),
                tf.keras.metrics.Recall(class_id=1, name='female_recall'),
                tf.keras.metrics.Precision(class_id=0, name='male_precision'),
                tf.keras.metrics.Recall(class_id=0, name='male_recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        history_ft1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best weights
        self.model.load_weights('best_model.weights.h5')
        
        # Fine-tuning phase 2: Train all layers
        print("\nPhase 3: Fine-tuning all layers...")
        base_model.trainable = True
        
        # Recompile with very low learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            loss=self._focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(class_id=1, name='female_precision'),
                tf.keras.metrics.Recall(class_id=1, name='female_recall'),
                tf.keras.metrics.Precision(class_id=0, name='male_precision'),
                tf.keras.metrics.Recall(class_id=0, name='male_recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        history_ft2 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best weights from all phases
        self.model.load_weights('best_model.weights.h5')
        
        return history
    
    def predict(self, image):
        """Predict gender with confidence threshold."""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        if image.shape[1:] != (64, 64, 3):
            image = tf.image.resize(image, (64, 64))
        
        # Get prediction probabilities
        pred_probs = self.model.predict(image)[0]
        male_prob, female_prob = pred_probs
        
        # Calculate confidence
        confidence = max(male_prob, female_prob)
        
        # Use dynamic threshold based on confidence
        threshold = 0.6  # Increased threshold for more confident predictions
        if confidence < threshold:
            print("\nWarning: Low confidence prediction")
            
        print("\nPrediction Details:")
        print(f"Male probability: {male_prob:.4f}")
        print(f"Female probability: {female_prob:.4f}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Threshold: {threshold}")
        
        if confidence < threshold:
            print("Prediction: Uncertain")
            return None
        else:
            result = female_prob > male_prob
            print(f"Predicted gender: {'female' if result else 'male'}")
            return result
    
    def evaluate(self, X_test, y_test):
        """Evaluate model with detailed metrics."""
        # Convert labels to categorical
        y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        total = len(y_test)
        correct = np.sum(y_pred == y_test)
        
        male_mask = (y_test == 0)
        female_mask = (y_test == 1)
        
        male_correct = np.sum((y_pred == y_test) & male_mask)
        female_correct = np.sum((y_pred == y_test) & female_mask)
        
        male_total = np.sum(male_mask)
        female_total = np.sum(female_mask)
        
        # Calculate confusion matrix metrics
        tp = np.sum((y_pred == 1) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        print("\nDetailed Evaluation Metrics:")
        print(f"Total Accuracy: {correct/total:.2%}")
        print(f"Male Accuracy: {male_correct/male_total:.2%} ({male_correct}/{male_total})")
        print(f"Female Accuracy: {female_correct/female_total:.2%} ({female_correct}/{female_total})")
        print("\nConfusion Matrix Metrics:")
        print(f"True Positives (Female): {tp}")
        print(f"True Negatives (Male): {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.2%}")
        print(f"Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.2%}")
        
        return {
            'total_accuracy': correct/total,
            'male_accuracy': male_correct/male_total if male_total > 0 else 0,
            'female_accuracy': female_correct/female_total if female_total > 0 else 0,
            'precision': tp/(tp+fp) if (tp+fp) > 0 else 0,
            'recall': tp/(tp+fn) if (tp+fn) > 0 else 0
        }
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = models.load_model(path) 