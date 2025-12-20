import os
from PIL import Image
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    nn = None

# Defer torchvision import until transforms are required
_CNN_TRANSFORMS = None

try:
    from .config import IMAGE_SIZE
except Exception:
    IMAGE_SIZE = 64


def _get_transforms():
    global _CNN_TRANSFORMS
    if _CNN_TRANSFORMS is None:
        try:
            from torchvision import transforms
            _CNN_TRANSFORMS = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        except Exception:
            _CNN_TRANSFORMS = None
    return _CNN_TRANSFORMS

# Define SimpleCNN only if torch is available
if torch is not None:
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool3 = nn.MaxPool2d(2, 2)
            ds = IMAGE_SIZE // 8
            self.flatten_size = 128 * ds * ds
            self.fc1 = nn.Linear(self.flatten_size, 512)
            self.fc2 = nn.Linear(512, num_classes)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


def prepare_cnn_input_data(filepaths, load_center_slice_func):
    transforms = _get_transforms()
    if transforms is None:
        print("PyTorch torchvision.transforms unavailable")
        return None
    processed_images = []
    if hasattr(filepaths, 'tolist'):
        filepaths = filepaths.tolist()
    for path in filepaths:
        slice_img = load_center_slice_func([path])
        if slice_img is None:
            continue
        try:
            img = slice_img.convert('L')
            tensor = transforms(img)
            processed_images.append(tensor)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    if not processed_images:
        return None
    X = torch.stack(processed_images)
    return X


class ImageDataset(Dataset):
    def __init__(self, X_data, y_labels):
        self.X = X_data
        self.y = y_labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_cnn_model_logic(X_data, y_labels, num_classes, output_dir, SimpleCNNClass=None,
                          NUM_EPOCHS=500, BATCH_SIZE=8, LR=0.001):
    if torch is None:
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    y_train_tensor = torch.from_numpy(y_train).long()
    y_val_tensor = torch.from_numpy(y_val).long()
    train_dataset = ImageDataset(X_train, y_train_tensor)
    val_dataset = ImageDataset(X_val, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_cls = SimpleCNNClass or SimpleCNN
    model = model_cls(num_classes=num_classes).to(device)
    model_path = os.path.join(output_dir, "cnn_classifier_model.pth")
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {running_loss / max(len(train_loader),1):.4f}, Val Acc: {epoch_accuracy:.2f}%")
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_path)
    return model_path, best_accuracy


def TrainImageClassifierLogic(output_dir, gui):
    """
    Aggregates classification CSVs, prepares CNN input data, encodes targets,
    and trains the CNN using train_cnn_model_logic. Updates the provided GUI
    (TrainClassifierGUI) with progress messages.
    """
    try:
        import pandas as pd
        import joblib
        from sklearn.preprocessing import LabelEncoder
        import time
        from . import processing as processing_module
    except Exception as e:
        try:
            gui.update_status("❌ Error: Missing dependencies for Image Model training.")
            time.sleep(2)
            gui.close()
        except Exception:
            pass
        return

    try:
        gui.update_status("1/6: Aggregating data for Image Model...")

        # Collect classification CSV files
        all_data = []
        for study_folder in os.listdir(output_dir):
            study_path = os.path.join(output_dir, study_folder)
            if os.path.isdir(study_path):
                csv_path = os.path.join(study_path, "classification_results.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

        if not all_data:
            gui.update_status("❌ Error: No classification CSV files found to train on.")
            time.sleep(2)
            gui.close()
            return

        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv(os.path.join(output_dir, "classification_data.csv"), index=False)

        # Remove 'other' and 'none' labels
        master_df = master_df[master_df['Classification'].str.lower() != 'other'].copy()
        master_df = master_df[master_df['Classification'].str.lower() != 'none'].copy()
        if master_df.empty:
            gui.update_status("❌ Error: Training data is empty after removing 'None'/'other' labels.")
            time.sleep(2)
            gui.close()
            return

        # Prepare image tensors
        gui.update_status("2/6: Preparing image data (center slices)...")
        dicom_paths = master_df['Path']
        X = prepare_cnn_input_data(dicom_paths, load_center_slice_func=processing_module.load_center_slice)
        if X is None:
            gui.update_status("❌ Error: Failed to prepare CNN input data (missing torchvision or no images).")
            time.sleep(2)
            gui.close()
            return

        y = master_df['Classification']

        # Load or fit target encoder
        try:
            target_encoder_filename = os.path.join(output_dir, "label_encoder.joblib")
            if os.path.exists(target_encoder_filename):
                target_encoder = joblib.load(target_encoder_filename)
                y_encoded = target_encoder.transform(y)
            else:
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y)
            num_classes = len(target_encoder.classes_)
        except Exception as e:
            gui.update_status(f"❌ Error preparing target encoder: {e}")
            time.sleep(2)
            gui.close()
            return

        gui.update_status(f"4/6: Training SimpleCNN (Num Classes: {num_classes})...")

        # Train the model (this is a blocking call; caller should run in a thread)
        try:
            model_path, best_accuracy = train_cnn_model_logic(X, y_encoded, num_classes, output_dir)
        except Exception as e:
            gui.update_status(f"❌ Error during CNN training: {e}")
            time.sleep(2)
            gui.close()
            return

        # Save the target encoder
        try:
            joblib.dump(target_encoder, target_encoder_filename)
        except Exception as e:
            print(f"Warning: failed to save target encoder: {e}")

        # Save a short summary
        try:
            final_message = f"✅ Image Model Training.\nModel Saved: {model_path}\nBest Accuracy: {best_accuracy:.2f}%"
            summary_path = os.path.join(output_dir, "cnn_model_training_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(final_message)
            gui.update_status(final_message)
            time.sleep(3)
        except Exception:
            pass

    except Exception as e:
        try:
            gui.update_status(f"❌ An unexpected error occurred during Image Model training: {type(e).__name__}: {e}")
            time.sleep(3)
        except Exception:
            pass

    finally:
        try:
            gui.close()
        except Exception:
            pass