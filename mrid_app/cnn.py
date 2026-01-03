import os
from PIL import Image
import numpy as np
import pickle
import hashlib
import multiprocessing as mp

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
_AUGMENTATION_TRANSFORMS = None

try:
    from .config import IMAGE_SIZE, NUM_EPOCHS, BATCH_SIZE, LR
except Exception:
    IMAGE_SIZE = 64
    NUM_EPOCHS = 200
    BATCH_SIZE = 8
    LR = 0.001


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


def _get_augmentation_transforms():
    """Get transforms for on-the-fly augmentation (random dimension swaps and rotations)"""
    global _AUGMENTATION_TRANSFORMS
    if _AUGMENTATION_TRANSFORMS is None:
        try:
            from torchvision import transforms
            _AUGMENTATION_TRANSFORMS = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        except Exception:
            _AUGMENTATION_TRANSFORMS = None
    return _AUGMENTATION_TRANSFORMS

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


def _process_single_image(args):
    """
    Worker function for parallel image preprocessing.

    Args:
        args: Tuple of (idx, path, label)

    Returns:
        Tuple of (idx, tensor, label) or (idx, None, None) if failed
    """
    idx, path, label = args

    try:
        # Import here to avoid pickling issues
        from . import processing as processing_module

        # Get transforms
        transforms = _get_transforms()
        if transforms is None:
            return (idx, None, None)

        # Load and process image
        slice_img = processing_module.load_center_slice([path])
        if slice_img is None:
            return (idx, None, None)

        img = slice_img.convert('L')
        tensor = transforms(img)

        return (idx, tensor, label)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return (idx, None, None)


def prepare_and_cache_cnn_data(filepaths, labels, load_center_slice_func, output_dir, progress_callback=None, num_workers=None):
    """
    Preprocess training data by applying standard transforms and save to cache for fast I/O.
    Uses parallel processing for faster preprocessing.
    Returns the cache file path if successful, None otherwise.

    Args:
        filepaths: List of file paths to process
        labels: List of labels corresponding to filepaths
        load_center_slice_func: Function to load center slice from DICOM
        output_dir: Directory to save cache file
        progress_callback: Optional callback function(current, total, message) for progress updates
        num_workers: Number of parallel workers (default: cpu_count - 1)
    """
    transforms = _get_transforms()
    if transforms is None:
        print("PyTorch torchvision.transforms unavailable")
        return None

    # Generate cache filename based on data hash and save directly to output_dir
    data_hash = hashlib.md5(str(sorted(filepaths)).encode()).hexdigest()[:16]
    cache_path = os.path.join(output_dir, f"cnn_preprocessed_data_{data_hash}.pkl")

    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading preprocessed data from cache: {cache_path}")
        if progress_callback:
            progress_callback(0, 1, "Loading from cache...")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            if progress_callback:
                progress_callback(1, 1, "Cache loaded successfully")
            return cache_path
        except Exception as e:
            print(f"Error loading cache, will regenerate: {e}")

    # Preprocess all images using parallel processing
    total_files = len(filepaths)
    print(f"Preprocessing {total_files} images using parallel processing...")
    if progress_callback:
        progress_callback(0, total_files, f"Preprocessing 0/{total_files} images...")

    if hasattr(filepaths, 'tolist'):
        filepaths = filepaths.tolist()
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Prepare arguments for parallel processing
    args_list = [(idx, path, label) for idx, (path, label) in enumerate(zip(filepaths, labels))]

    # Process images in parallel with progress tracking
    processed_results = []
    completed = 0

    try:
        # Use Pool with imap_unordered for progress tracking
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_process_single_image, args_list, chunksize=1):
                completed += 1

                # Only keep successful results
                if result[1] is not None:
                    processed_results.append(result)

                # Update progress every 10 images or on last image
                if progress_callback and (completed % 10 == 0 or completed == total_files):
                    progress_callback(completed, total_files, f"Preprocessing {completed}/{total_files} images...")

    except Exception as e:
        print(f"Error during parallel processing: {e}")
        # Fall back to sequential processing
        print("Falling back to sequential processing...")
        processed_results = []
        for idx, path in enumerate(filepaths):
            slice_img = load_center_slice_func([path])
            if slice_img is None:
                continue
            try:
                img = slice_img.convert('L')
                tensor = transforms(img)
                processed_results.append((idx, tensor, labels[idx]))
            except Exception as e:
                print(f"Error processing {path}: {e}")

            # Update progress every 10 images or on last image
            if progress_callback and (idx % 10 == 0 or idx == total_files - 1):
                progress_callback(idx + 1, total_files, f"Preprocessing {idx + 1}/{total_files} images...")

    if not processed_results:
        print("No images were successfully processed")
        return None

    # Sort results by original index to maintain order
    processed_results.sort(key=lambda x: x[0])

    # Extract tensors and labels
    processed_images = [result[1] for result in processed_results]
    valid_labels = [result[2] for result in processed_results]

    # Stack tensors and save to cache
    if progress_callback:
        progress_callback(total_files, total_files, "Saving preprocessed data to cache...")

    X = torch.stack(processed_images)
    y = np.array(valid_labels)

    cached_data = {'X': X, 'y': y}

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"Cached preprocessed data to: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return cache_path


def load_cached_data(cache_path):
    """Load preprocessed data from cache file."""
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data['X'], cached_data['y']
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None, None


class ImageDataset(Dataset):
    def __init__(self, X_data, y_labels):
        self.X = X_data
        self.y = y_labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AugmentedImageDataset(Dataset):
    """
    Dataset with on-the-fly augmentation using random dimension swaps and rotations.
    Applied to preprocessed, cached training data.
    """
    def __init__(self, X_data, y_labels, apply_augmentation=True):
        self.X = X_data
        self.y = y_labels
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.apply_augmentation and torch is not None:
            # Random dimension swaps (transpose)
            if np.random.rand() > 0.5:
                # Swap height and width dimensions (dims 1 and 2 for CHW format)
                image = torch.transpose(image, 1, 2)

            # Random rotations (90, 180, 270 degrees)
            k = np.random.randint(0, 4)  # 0, 1, 2, or 3 rotations of 90 degrees
            if k > 0:
                # Rotate in the spatial dimensions (dims 1 and 2)
                image = torch.rot90(image, k=k, dims=[1, 2])

            # Random flips
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=[1])  # Horizontal flip
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=[2])  # Vertical flip

        return image, label


def train_cnn_model_logic(X_data, y_labels, num_classes, output_dir, SimpleCNNClass=None, progress_callback=None):
    """
    Train CNN model with optional progress reporting.

    Args:
        X_data: Training data
        y_labels: Training labels
        num_classes: Number of classes
        output_dir: Directory to save model
        SimpleCNNClass: Optional custom CNN class
        progress_callback: Optional callback function(current_epoch, total_epochs, loss, accuracy, message)
    """
    if torch is None:
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    y_train_tensor = torch.from_numpy(y_train).long()
    y_val_tensor = torch.from_numpy(y_val).long()

    # Use AugmentedImageDataset for training (with augmentation)
    # Use regular ImageDataset for validation (no augmentation)
    train_dataset = AugmentedImageDataset(X_train, y_train_tensor, apply_augmentation=True)
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
        avg_loss = running_loss / max(len(train_loader), 1)

        # Progress callback
        if progress_callback:
            message = f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}, Val Acc: {epoch_accuracy:.2f}%"
            progress_callback(epoch + 1, NUM_EPOCHS, avg_loss, epoch_accuracy, message)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}, Val Acc: {epoch_accuracy:.2f}%")

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
                study_id = os.path.basename(study_path.rstrip(os.sep))
                csv_path = os.path.join(study_path, f"{study_id}_classification-results.csv")
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

        # Set up progress callback for preprocessing
        total_paths = len(dicom_paths)
        gui.set_determinate_mode(total_paths)

        def preprocessing_progress(current, total, message):
            """Callback to update GUI during preprocessing"""
            gui.update_progress(current)
            gui.update_status(f"2/6: {message}")

        # Try to load from cache first, or preprocess and cache
        cache_path = prepare_and_cache_cnn_data(
            dicom_paths.tolist(),
            master_df['Classification'].tolist(),
            load_center_slice_func=processing_module.load_center_slice,
            output_dir=output_dir,
            progress_callback=preprocessing_progress
        )

        # Switch back to indeterminate mode
        gui.set_indeterminate_mode()

        if cache_path is None:
            gui.update_status("❌ Error: Failed to prepare CNN input data (missing torchvision or no images).")
            time.sleep(2)
            gui.close()
            return

        # Load cached data
        gui.update_status("3/6: Loading preprocessed data from cache...")
        X, y_str = load_cached_data(cache_path)
        if X is None:
            gui.update_status("❌ Error: Failed to load cached data.")
            time.sleep(2)
            gui.close()
            return

        # Note: y_str contains the string labels, need to filter and encode them
        # Filter to match only valid classifications
        valid_indices = []
        valid_classifications = []
        for idx, classification in enumerate(y_str):
            if str(classification).lower() not in ['other', 'none']:
                valid_indices.append(idx)
                valid_classifications.append(classification)

        if not valid_indices:
            gui.update_status("❌ Error: No valid training samples after filtering.")
            time.sleep(2)
            gui.close()
            return

        X = X[valid_indices]
        y = valid_classifications

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

        # Set up progress callback for training
        gui.set_determinate_mode(NUM_EPOCHS)

        def training_progress(current_epoch, total_epochs, loss, accuracy, message):
            """Callback to update GUI during training"""
            gui.update_progress(current_epoch)
            gui.update_status(f"4/6: {message}")

        # Train the model (this is a blocking call; caller should run in a thread)
        try:
            model_path, best_accuracy = train_cnn_model_logic(
                X, y_encoded, num_classes, output_dir,
                progress_callback=training_progress
            )
        except Exception as e:
            gui.update_status(f"❌ Error during CNN training: {e}")
            time.sleep(2)
            gui.close()
            return

        # Switch back to indeterminate mode
        gui.set_indeterminate_mode()

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