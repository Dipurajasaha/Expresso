import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Import custom modules
from model import create_model
from data_preprocessing import get_data_generators, balance_dataset

# --- Configuration & Constants ---
BASE_DATA_DIR = '../data/Dataset'
BALANCED_DATA_DIR = '../data/dataset_balanced'
TRAIN_DIR_ORIGINAL = os.path.join(BASE_DATA_DIR, 'train')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')

TRAIN_DIR_BALANCED = os.path.join(BALANCED_DATA_DIR, 'train')

MODEL_DIR = '../saved_models'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')
BATCH_SIZE = 64
EPOCHS = 100

os.makedirs(MODEL_DIR, exist_ok=True)


def plot_training_history(history):
    """Plots accuracy and loss curves for training and validation sets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()

def main():
    # --- 1. Balance Dataset (run once) ---
    if not os.path.exists(TRAIN_DIR_BALANCED) or not os.listdir(TRAIN_DIR_BALANCED):
        print("Balanced training data not found. Running the balancing script...")
        balance_dataset(TRAIN_DIR_ORIGINAL, TRAIN_DIR_BALANCED)
    else:
        print("Balanced training data found. Skipping balancing step.")

    # --- 2. Load Data Generators ---
    train_generator, test_generator = get_data_generators(TRAIN_DIR_BALANCED, TEST_DIR, BATCH_SIZE)

    # --- 3. Build or Load Model ---
    if os.path.exists(BEST_MODEL_PATH):
        print("\nResuming from saved model...\n")
        model = load_model(BEST_MODEL_PATH)
    else:
        print("\nStarting from scratch...\n")
        model = create_model()
    model.summary()

    # --- 4. Callbacks ---
    checkpoint_model = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

    # --- 5. Train Model ---
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=[early_stopping, lr_scheduler, checkpoint_model]
    )

    # --- 6. Plot History and Evaluate ---
    plot_training_history(history)
    
    # Load the best model saved by the checkpoint
    best_model = load_model(BEST_MODEL_PATH)
    
    print("\n--- Final Evaluation on Test Set ---")
    loss, accuracy = best_model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # --- 7. Classification Report and Confusion Matrix ---
    y_pred_probs = best_model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    main()