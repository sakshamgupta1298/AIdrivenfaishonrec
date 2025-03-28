# Gender Detection and Fashion Recommendation System

An AI-powered system that detects gender from facial features and provides fashion recommendations based on the detected gender and occasion.

## Features

- Gender detection using deep learning (CNN)
- Fashion recommendations for different occasions
- Support for multiple image formats
- Real-time processing capabilities
- Interactive testing interface
- Detailed model evaluation metrics

## Project Structure

```
├── gender_detection/       # Gender detection model
├── fashion_recommender/    # Fashion recommendation system
├── data/                  # Data directory
│   └── dataset/          # Training and testing dataset
│       ├── male/        # Male images
│       └── female/      # Female images
├── main.py               # Main application
├── train_model.py        # Model training script
├── evaluate_model.py     # Model evaluation script
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd gender-detection-fashion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train_model.py
```

2. Evaluate model performance:
```bash
python evaluate_model.py
```

3. Run the main application:
```bash
python main.py
```

## Model Architecture

The gender detection model uses a Convolutional Neural Network (CNN) with:
- Multiple convolutional layers for feature extraction
- Batch normalization for training stability
- Dropout layers to prevent overfitting
- Dense layers for final classification

## Dataset Requirements

The model expects images organized in the following structure:
```
data/
  dataset/
    male/
      [male images]
    female/
      [female images]
```

- Supported image formats: JPG, JPEG, PNG
- Recommended image size: Any (will be resized to 64x64)
- Clear facial features should be visible

## Fashion Recommendations

The system provides outfit recommendations for:
- Casual occasions
- Formal events
- Party wear

## Evaluation Metrics

The evaluation script provides:
- Overall accuracy
- Per-class accuracy (male/female)
- Confusion matrix
- Detailed classification report

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for image processing
- TensorFlow for deep learning
- scikit-learn for evaluation metrics 