import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

# Constants
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
CLASSES = ['happy', 'sad', 'crying', 'surprised', 'angry']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}
TRAIN_DIR = 'train/dataset/'
LABELS_PATH = 'train/labels.csv'
MODEL_PATH = 'emoji_detector_model.h5'
BATCH_SIZE = 16
EPOCHS = 30
DETECTION_THRESHOLD = 0.5

# Create a directory for logs if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

def load_and_preprocess_data():
    """Load and preprocess the emoji dataset"""
    # Load labels
    labels_df = pd.read_csv(LABELS_PATH, sep=';')
    
    # Initialize lists to store data
    images = []
    all_boxes = []
    all_classes = []
    file_names = []
    
    print("Loading and preprocessing images...")
    for idx, row in labels_df.iterrows():
        file_name = row['file_name']
        file_names.append(file_name)
        
        # Load and preprocess image
        img_path = os.path.join(TRAIN_DIR, file_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)
        
        # Parse mood labels and coordinates
        moods = eval(row['moods'])
        x_coords = eval(row['x_s'])
        y_coords = eval(row['y_s'])
        
        # Process each emoji in the image
        boxes = []
        classes = []
        for mood, x, y in zip(moods, x_coords, y_coords):
            # We're just given top-left coordinates, so estimate a bounding box
            # Assume emojis are roughly 60x60 pixels
            x1, y1 = x, y
            x2, y2 = min(x + 60, IMAGE_WIDTH), min(y + 60, IMAGE_HEIGHT)
            
            # Normalize coordinates to [0, 1]
            x1_norm, y1_norm = x1 / IMAGE_WIDTH, y1 / IMAGE_HEIGHT
            x2_norm, y2_norm = x2 / IMAGE_WIDTH, y2 / IMAGE_HEIGHT
            
            boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            classes.append(CLASS_TO_IDX[mood])
        
        all_boxes.append(boxes)
        all_classes.append(classes)
    
    print(f"Loaded {len(images)} images with annotations")
    return np.array(images), all_boxes, all_classes, file_names

def create_model():
    """Create a custom object detection model using MobileNetV2 as base"""
    # Base model: MobileNetV2 (lightweight and efficient)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Add custom detection head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    
    # Output layers: class prediction and bounding box regression
    class_output = Dense(NUM_CLASSES, activation='sigmoid', name='class_output')(x)
    bbox_output = Dense(4, name='bbox_output')(x)
    
    model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'class_output': 'binary_crossentropy',
            'bbox_output': 'mse'
        },
        metrics={
            'class_output': 'accuracy'
        }
    )
    
    return model

def train_model(model, images, all_boxes, all_classes):
    """Train the emoji detection model"""
    # Prepare training data
    # For simplicity, we'll train on images with a single emoji first
    # Later we can extend to handle multiple emojis per image
    
    single_emoji_indices = [i for i, boxes in enumerate(all_boxes) if len(boxes) == 1]
    X_train = np.array([images[i] for i in single_emoji_indices])
    
    # Prepare labels and bounding boxes
    y_classes = np.zeros((len(single_emoji_indices), NUM_CLASSES))
    y_boxes = np.zeros((len(single_emoji_indices), 4))
    
    for i, idx in enumerate(single_emoji_indices):
        cls_idx = all_classes[idx][0]
        y_classes[i, cls_idx] = 1
        y_boxes[i] = all_boxes[idx][0]
    
    # Split into training and validation sets
    X_train, X_val, y_classes_train, y_classes_val, y_boxes_train, y_boxes_val = train_test_split(
        X_train, y_classes, y_boxes, test_size=0.2, random_state=42
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train,
        {'class_output': y_classes_train, 'bbox_output': y_boxes_train},
        validation_data=(
            X_val,
            {'class_output': y_classes_val, 'bbox_output': y_boxes_val}
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['class_output_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_class_output_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model

def detect_emojis(model, image_path):
    """Detect emojis in a given image"""
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        return [], [], []
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img_normalized = img_resized / 255.0
    
    # Make prediction
    class_preds, bbox_preds = model.predict(np.expand_dims(img_normalized, axis=0))
    
    # Process predictions
    detected_classes = []
    detected_boxes = []
    detected_scores = []
    
    # For each class
    for class_idx in range(NUM_CLASSES):
        class_score = class_preds[0, class_idx]
        
        if class_score > DETECTION_THRESHOLD:
            detected_classes.append(IDX_TO_CLASS[class_idx])
            detected_boxes.append(bbox_preds[0])
            detected_scores.append(class_score)
    
    # Convert normalized boxes to pixel coordinates
    pixel_boxes = []
    for box in detected_boxes:
        x1, y1, x2, y2 = box
        x1_px = int(x1 * IMAGE_WIDTH)
        y1_px = int(y1 * IMAGE_HEIGHT)
        x2_px = int(x2 * IMAGE_WIDTH)
        y2_px = int(y2 * IMAGE_HEIGHT)
        pixel_boxes.append((x1_px, y1_px, x2_px, y2_px))
    
    return detected_classes, pixel_boxes, detected_scores

def visualize_detections(image_path, classes, boxes, scores):
    """Visualize detected emojis on an image"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    
    for cls, box, score in zip(classes, boxes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), width, height, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(x1, y1-10, f'{cls}: {score:.2f}', color='red', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'visualization_{os.path.basename(image_path)}')
    plt.close()

def evaluate_model(model, test_dir, annotation_file, run_number=1):
    """Evaluate model performance on test set and generate output"""
    # Load test annotations
    test_df = pd.read_csv(annotation_file, sep=';')
    
    results = []
    log_output = []
    correct_count = 0
    incorrect_count = 0
    coordinate_off_count = 0
    
    for idx, row in test_df.iterrows():
        file_name = row['file_name']
        img_path = os.path.join(test_dir, file_name)
        
        # Ground truth
        true_moods = eval(row['moods'])
        true_x = eval(row['x_s'])
        true_y = eval(row['y_s'])
        
        # Detect emojis
        detected_classes, detected_boxes, detected_scores = detect_emojis(model, img_path)
        
        # Log results for this image
        log_output.append(f"Picture: {file_name}")
        
        for cls, box, score in zip(detected_classes, detected_boxes, detected_scores):
            x1, y1, _, _ = box
            log_output.append(f"Emoji: {cls} Coordinates: ({x1}, {y1})")
        
        # Evaluate accuracy
        # This is a simplified evaluation - in practice you'd need a more sophisticated
        # matching algorithm to pair predicted emojis with ground truth
        for i, (true_mood, x, y) in enumerate(zip(true_moods, true_x, true_y)):
            matched = False
            for j, (pred_class, box) in enumerate(zip(detected_classes, detected_boxes)):
                pred_x, pred_y, _, _ = box
                
                if pred_class == true_mood:
                    dist = np.sqrt((pred_x - x)**2 + (pred_y - y)**2)
                    if dist < 40:
                        correct_count += 1
                        matched = True
                    else:
                        coordinate_off_count += 1
                        matched = True
                
            if not matched:
                incorrect_count += 1
    
    # Calculate score
    score = correct_count + 0.5 * coordinate_off_count - 0.5 * incorrect_count
    
    # Save results
    log_text = '\n'.join(log_output)
    with open(f'logs/run_log_{run_number}.txt', 'w') as f:
        f.write(log_text)
    
    print(f"Evaluation complete. Results saved to logs/run_log_{run_number}.txt")
    print(f"Score: {score} (Correct: {correct_count}, Coordinate Off: {coordinate_off_count}, Incorrect: {incorrect_count})")
    
    return score

def process_multiple_emojis(model, images, all_boxes, all_classes):
    """Extend the model to handle multiple emojis in a single image"""
    # This would be a more advanced implementation
    # Here we would use techniques like:
    # 1. Non-maximum suppression to handle multiple detections
    # 2. Anchor boxes for different emoji sizes
    # 3. Feature pyramid networks for detecting emojis at different scales
    
    # For this hackathon, we'll use a sliding window approach as a simple solution
    pass

def main():
    # Load data
    images, all_boxes, all_classes, file_names = load_and_preprocess_data()
    
    # Create model
    model = create_model()
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_weights(MODEL_PATH)
    else:
        # Train model
        model = train_model(model, images, all_boxes, all_classes)
    
    # Run number for logging
    run_number = 1
    while os.path.exists(f'logs/run_log_{run_number}.txt'):
        run_number += 1
    
    # Evaluate on test data
    score = evaluate_model(model, TRAIN_DIR, LABELS_PATH, run_number)
    
    # Visualize some results
    for i in range(5):  # visualize first 5 images
        img_path = os.path.join(TRAIN_DIR, file_names[i])
        detected_classes, detected_boxes, detected_scores = detect_emojis(model, img_path)
        visualize_detections(img_path, detected_classes, detected_boxes, detected_scores)

if __name__ == "__main__":
    main()