import os
from PIL import Image
import numpy as np
import pickle
import hashlib
import multiprocessing as mp
from .utils import (extract_study_id_from_path, get_study_csv_name,
                    get_classified_outputs_dir, get_num_parallel_workers)
from .logging_config import get_logger

logger = get_logger()

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    nn = None

# Defer torchvision import until transforms are required
_CNN_TRANSFORMS = {}  # image_size -> transform pipeline
_AUGMENTATION_TRANSFORMS = None

# Fallback defaults; live values are read from the config module at call time
# (via _cfg) so configs loaded at runtime are respected.
try:
    from .config import IMAGE_SIZE, NUM_EPOCHS, BATCH_SIZE, LR
except Exception:
    IMAGE_SIZE = 64
    NUM_EPOCHS = 200
    BATCH_SIZE = 8
    LR = 0.001


def _cfg(name, default):
    """Read a config value at call time so runtime config loads/reloads apply."""
    try:
        from . import config as _config
        return getattr(_config, name, default)
    except Exception:
        return default


def get_configured_num_slices():
    """CNN_NUM_SLICES at call time. 1 = 2D (center slice), >1 = 2.5D."""
    try:
        return max(1, int(_cfg('CNN_NUM_SLICES', 1)))
    except Exception:
        return 1


def get_torch_device():
    """Best available torch device: CUDA (NVIDIA/AMD ROCm) > Apple MPS > CPU."""
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def _get_transforms(image_size=None):
    if image_size is None:
        image_size = int(_cfg('IMAGE_SIZE', IMAGE_SIZE))
    if image_size not in _CNN_TRANSFORMS:
        try:
            from torchvision import transforms
            _CNN_TRANSFORMS[image_size] = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        except Exception:
            return None
    return _CNN_TRANSFORMS[image_size]


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
        def __init__(self, num_classes, num_slices=None, image_size=None):
            super(SimpleCNN, self).__init__()
            # Input channels = number of slices (1 = 2D, >1 = 2.5D)
            self.num_slices = num_slices if num_slices is not None else get_configured_num_slices()
            self.image_size = image_size if image_size is not None else int(_cfg('IMAGE_SIZE', IMAGE_SIZE))
            self.conv1 = nn.Conv2d(self.num_slices, 32, 3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool3 = nn.MaxPool2d(2, 2)
            ds = self.image_size // 8
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


def load_cnn_model(model_path, num_classes, map_location=None):
    """
    Load a saved SimpleCNN checkpoint. The number of input slices and the
    image size are inferred from the checkpoint weights so a saved model
    loads and runs correctly regardless of the current config values.

    Returns the model in eval mode, or raises on failure.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    state_dict = torch.load(model_path, map_location=map_location)
    try:
        ckpt_slices = int(state_dict['conv1.weight'].shape[1])
    except Exception:
        ckpt_slices = 1
    try:
        # fc1 in_features = 128 * (image_size // 8)^2
        in_features = int(state_dict['fc1.weight'].shape[1])
        ckpt_image_size = int((in_features // 128) ** 0.5) * 8
    except Exception:
        ckpt_image_size = None
    configured = get_configured_num_slices()
    if ckpt_slices != configured:
        logger.warning(
            f"Saved CNN model was trained with CNN_NUM_SLICES={ckpt_slices} but the "
            f"config specifies {configured}. Using {ckpt_slices} for inference; "
            f"retrain the model to change the slice count."
        )
    model = SimpleCNN(num_classes=num_classes, num_slices=ckpt_slices,
                      image_size=ckpt_image_size)
    model.load_state_dict(state_dict)
    if map_location is not None:
        model.to(map_location)
    model.eval()
    return model


def _slices_to_array(slice_imgs, image_size):
    """
    Convert a list of PIL slice images to a (num_slices, H, W) float32 array
    normalized to [-1, 1] (matching ToTensor + Normalize(0.5, 0.5)).
    """
    from PIL import Image as PILImage
    channels = []
    for slice_img in slice_imgs:
        img = slice_img.convert('L').resize((image_size, image_size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        channels.append(arr)
    return np.stack(channels, axis=0)


def prepare_cnn_input_data(filepaths, load_slices_func, num_slices=None, image_size=None):
    transforms = _get_transforms(image_size)
    if transforms is None:
        logger.warning("PyTorch torchvision.transforms unavailable")
        return None
    if num_slices is None:
        num_slices = get_configured_num_slices()
    processed_images = []
    if hasattr(filepaths, 'tolist'):
        filepaths = filepaths.tolist()
    for path in filepaths:
        slice_imgs = load_slices_func([path], num_slices)
        if not slice_imgs:
            continue
        try:
            # Transform each slice separately, then stack as channels
            channels = [transforms(img.convert('L')) for img in slice_imgs]
            tensor = torch.cat(channels, dim=0)
            processed_images.append(tensor)
        except Exception as e:
            logger.debug(f"Error processing {path}: {e}")
    if not processed_images:
        return None
    X = torch.stack(processed_images)
    return X


def _process_single_image(args):
    """
    Worker function for parallel image preprocessing.

    Args:
        args: Tuple of (idx, path, label, image_size, num_slices)

    Returns:
        Tuple of (idx, numpy_array, label) or (idx, None, None) if failed.
        Returns a numpy array (not a tensor) to avoid passing PyTorch shared-memory
        file descriptors through the multiprocessing pipe, which causes
        "received 0 items of ancdata" errors on Linux.
    """
    idx, path, label, image_size, num_slices = args

    try:
        # Import here to avoid pickling issues
        from . import processing as processing_module

        slice_imgs = processing_module.load_series_slices([path], num_slices)
        if not slice_imgs:
            return (idx, None, None)

        arr = _slices_to_array(slice_imgs, image_size)
        return (idx, arr, label)
    except Exception as e:
        logger.debug(f"Error processing {path}: {e}")
        return (idx, None, None)


def prepare_and_cache_cnn_data(filepaths, labels, load_slices_func, output_dir, progress_callback=None, num_workers=None, num_slices=None, cancel_check=None):
    """
    Preprocess training data by applying standard transforms and save to cache for fast I/O.
    Uses parallel processing for faster preprocessing.
    Returns the cache file path if successful, None otherwise (including if the
    user cancelled — callers should check cancel state to distinguish).

    Args:
        filepaths: List of file paths to process
        labels: List of labels corresponding to filepaths
        load_slices_func: Function(paths, num_slices) loading slices from a DICOM series
        output_dir: Directory to save cache file
        progress_callback: Optional callback function(current, total, message) for progress updates
        num_workers: Number of parallel workers (default: config.NUM_PARALLEL_WORKERS)
        num_slices: Slices per series fed as CNN channels (default: CNN_NUM_SLICES)
        cancel_check: Optional callable returning True to abort; checked during
            preprocessing so cancellation takes effect promptly (worker pool is
            terminated immediately).
    """
    transforms = _get_transforms()
    if transforms is None:
        logger.warning("PyTorch torchvision.transforms unavailable")
        return None

    if num_slices is None:
        num_slices = get_configured_num_slices()

    # Generate cache filename based on data hash and save to model_files subdirectory
    model_files_dir = os.path.join(output_dir, "model_files")
    os.makedirs(model_files_dir, exist_ok=True)

    data_hash = hashlib.md5(str(sorted(filepaths)).encode()).hexdigest()[:16]
    # Keep the legacy cache filename for 2D mode so existing caches remain valid
    slice_suffix = "" if num_slices == 1 else f"_{num_slices}slice"
    cache_path = os.path.join(model_files_dir, f"cnn_preprocessed_data_{data_hash}{slice_suffix}.pkl")

    if os.path.exists(cache_path):
        logger.info(f"Loading preprocessed data from cache: {cache_path}")
        if progress_callback:
            progress_callback(0, 1, "Loading from cache...")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            if progress_callback:
                progress_callback(1, 1, "Cache loaded successfully")
            return cache_path
        except Exception as e:
            logger.warning(f"Error loading cache, will regenerate: {e}")

    # Preprocess all images using parallel processing
    total_files = len(filepaths)
    logger.info(f"Preprocessing {total_files} images using parallel processing...")
    if progress_callback:
        progress_callback(0, total_files, f"Preprocessing 0/{total_files} images...")

    if hasattr(filepaths, 'tolist'):
        filepaths = filepaths.tolist()
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()

    # Number of workers follows config.NUM_PARALLEL_WORKERS (read live via the
    # shared helper); an explicit num_workers argument overrides it.
    if num_workers is None:
        num_workers = get_num_parallel_workers()
    else:
        num_workers = max(1, num_workers)

    # Use a chunksize proportional to the dataset size to reduce IPC overhead.
    chunksize = max(1, total_files // (num_workers * 10))

    # Prepare arguments for parallel processing — include the image size so the
    # worker never needs to import torchvision transforms (avoids pickling issues).
    image_size = int(_cfg('IMAGE_SIZE', IMAGE_SIZE))
    args_list = [(idx, path, label, image_size, num_slices) for idx, (path, label) in enumerate(zip(filepaths, labels))]

    # Process images in parallel with progress tracking.
    # Workers return plain numpy arrays (no PyTorch tensors / shared-memory fds),
    # which are safe to pass through the pool pipe without ancdata limits.
    processed_results = []
    completed = 0

    try:
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_process_single_image, args_list, chunksize=chunksize):
                # Abort promptly on cancel: kill the worker processes and stop.
                if cancel_check is not None and cancel_check():
                    logger.info("CNN preprocessing cancelled by user.")
                    pool.terminate()
                    pool.join()
                    return None
                completed += 1

                if result[1] is not None:
                    processed_results.append(result)

                # Update progress every 10 images or on last image
                if progress_callback and (completed % 10 == 0 or completed == total_files):
                    progress_callback(completed, total_files, f"Preprocessing {completed}/{total_files} images...")

    except Exception as e:
        logger.warning(f"Error during parallel processing: {e}")
        # Fall back to sequential processing
        logger.info("Falling back to sequential processing...")
        processed_results = []
        for idx, path in enumerate(filepaths):
            if cancel_check is not None and cancel_check():
                logger.info("CNN preprocessing cancelled by user.")
                return None
            slice_imgs = load_slices_func([path], num_slices)
            if not slice_imgs:
                continue
            try:
                arr = _slices_to_array(slice_imgs, image_size)
                processed_results.append((idx, arr, labels[idx]))
            except Exception as ex:
                logger.debug(f"Error processing {path}: {ex}")

            # Update progress every 10 images or on last image
            if progress_callback and (idx % 10 == 0 or idx == total_files - 1):
                progress_callback(idx + 1, total_files, f"Preprocessing {idx + 1}/{total_files} images...")

    # A late cancel (e.g. just as the loop finished) should still abort.
    if cancel_check is not None and cancel_check():
        logger.info("CNN preprocessing cancelled by user.")
        return None

    if not processed_results:
        logger.error("No images were successfully processed")
        return None

    # Sort results by original index to maintain order
    processed_results.sort(key=lambda x: x[0])

    # Convert numpy arrays → tensors in the main process (safe: no IPC involved)
    processed_arrays = [result[1] for result in processed_results]
    valid_labels = [result[2] for result in processed_results]

    # Stack tensors and save to cache
    if progress_callback:
        progress_callback(total_files, total_files, "Saving preprocessed data to cache...")

    X = torch.from_numpy(np.stack(processed_arrays))
    y = np.array(valid_labels)

    cached_data = {'X': X, 'y': y}

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        logger.info(f"Cached preprocessed data to: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return cache_path


def load_cached_data(cache_path):
    """Load preprocessed data from cache file."""
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data['X'], cached_data['y']
    except Exception as e:
        logger.error(f"Error loading cached data: {e}")
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


def train_cnn_model_logic(X_data, y_labels, num_classes, output_dir, SimpleCNNClass=None, progress_callback=None, cancel_check=None):
    """
    Train CNN model with optional progress reporting.

    Args:
        X_data: Training data
        y_labels: Training labels
        num_classes: Number of classes
        output_dir: Directory to save model
        SimpleCNNClass: Optional custom CNN class
        progress_callback: Optional callback function(current_epoch, total_epochs, loss, accuracy, message)
        cancel_check: Optional callable returning True if training should be aborted
    """
    if torch is None:
        return None, None
    device = get_torch_device()
    logger.info(f"CNN training device: {device}")

    # Hyperparameters read at call time so runtime-loaded configs apply
    num_epochs = int(_cfg('NUM_EPOCHS', NUM_EPOCHS))
    batch_size = int(_cfg('BATCH_SIZE', BATCH_SIZE))
    lr = float(_cfg('LR', LR))

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    y_train_tensor = torch.from_numpy(y_train).long()
    y_val_tensor = torch.from_numpy(y_val).long()

    # Use AugmentedImageDataset for training (with augmentation)
    # Use regular ImageDataset for validation (no augmentation)
    train_dataset = AugmentedImageDataset(X_train, y_train_tensor, apply_augmentation=True)
    val_dataset = ImageDataset(X_val, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Channel count and image size come from the training data itself:
    # (B, num_slices, H, W)
    if hasattr(X_data, 'shape') and len(X_data.shape) == 4:
        num_slices = int(X_data.shape[1])
        image_size = int(X_data.shape[2])
    else:
        num_slices = 1
        image_size = None
    model_cls = SimpleCNNClass or SimpleCNN
    try:
        model = model_cls(num_classes=num_classes, num_slices=num_slices,
                          image_size=image_size).to(device)
    except TypeError:
        # Custom CNN classes may not accept num_slices/image_size
        model = model_cls(num_classes=num_classes).to(device)

    model_files_dir = os.path.join(output_dir, "model_files")
    os.makedirs(model_files_dir, exist_ok=True)
    model_path = os.path.join(model_files_dir, "cnn_classifier_model.pth")

    if os.path.isfile(model_path):
        # Resume from an existing checkpoint only if it is shape-compatible
        # (e.g. it was trained with the same CNN_NUM_SLICES and classes)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Resuming training from existing model checkpoint")
        except Exception as e:
            logger.warning(f"Existing checkpoint is incompatible (training from scratch): {e}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Check for cancellation at the start of each epoch
        if cancel_check is not None and cancel_check():
            logger.info("CNN training cancelled by user during epoch loop.")
            return model_path if best_accuracy > 0 else None, best_accuracy

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

        if progress_callback:
            message = f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Val Acc: {epoch_accuracy:.2f}%"
            progress_callback(epoch + 1, num_epochs, avg_loss, epoch_accuracy, message)

        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Val Acc: {epoch_accuracy:.2f}%")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_path)

    return model_path, best_accuracy


def TrainImageClassifierLogic(output_dir, gui, preloaded_df=None):
    """
    Aggregates classification CSVs, prepares CNN input data, encodes targets,
    and trains the CNN using train_cnn_model_logic. Updates the provided GUI
    (TrainClassifierGUI) with progress messages.

    Args:
        output_dir: Output directory (used for model_files and logging)
        gui: TrainClassifierGUI instance
        preloaded_df: Optional pre-loaded DataFrame to use instead of aggregating
                      per-study CSVs. When provided, CSV scanning is skipped.
    """
    from .logging_config import update_log_file_location

    # Set up file logging now that we have the output directory
    os.makedirs(output_dir, exist_ok=True)
    update_log_file_location(output_dir)

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
        if preloaded_df is not None:
            # ── Analyze mode: use the pre-loaded self-supervised DataFrame ────
            gui.update_status("1/6: Using self-supervised classification data for Image Model…")
            master_df = preloaded_df.copy()
            model_files_dir = os.path.join(output_dir, "model_files")
            os.makedirs(model_files_dir, exist_ok=True)
        else:
            # ── Normal mode: aggregate per-study classification CSVs ──────────
            gui.update_status("1/6: Aggregating data for Image Model...")

            # Collect classification CSV files
            all_data = []
            classified_dir = get_classified_outputs_dir(output_dir)
            scan_root = classified_dir if os.path.isdir(classified_dir) else output_dir
            for study_folder in os.listdir(scan_root):
                study_path = os.path.join(scan_root, study_folder)
                if os.path.isdir(study_path):
                    study_id = extract_study_id_from_path(study_path)
                    csv_path = os.path.join(study_path, get_study_csv_name(study_id))
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            all_data.append(df)
                        except Exception as e:
                            logger.warning(f"Error reading {csv_path}: {e}")

            if not all_data:
                gui.update_status("❌ Error: No classification CSV files found to train on.")
                time.sleep(2)
                gui.close()
                return

            master_df = pd.concat(all_data, ignore_index=True)

            model_files_dir = os.path.join(output_dir, "model_files")
            os.makedirs(model_files_dir, exist_ok=True)

            master_df.to_csv(os.path.join(model_files_dir, "classification_data.csv"), index=False)

        # Normally drop the negative 'other'/'none' labels so the model
        # discriminates among target series. But with only ONE target series,
        # dropping them leaves a single class and softmax returns 1.0 for every
        # series (no discrimination). In that case keep the negatives as a single
        # 'Other' class so the model can score target-vs-Other.
        cls_lower = master_df['Classification'].str.lower()
        non_neg = master_df[(cls_lower != 'none') & (cls_lower != 'other')]
        if non_neg['Classification'].nunique() >= 2:
            master_df = non_neg.copy()
        else:
            master_df = master_df.copy()
            master_df.loc[cls_lower.isin(['none', 'other']), 'Classification'] = 'Other'
            logger.info("Only one target series — training target-vs-'Other' so the "
                        "model can discriminate (kept negative examples).")
        if master_df.empty:
            gui.update_status("❌ Error: Training data is empty after removing 'None'/'other' labels.")
            time.sleep(2)
            gui.close()
            return
        if master_df['Classification'].nunique() < 2:
            gui.update_status("❌ Error: Need at least two classes to train. With a single "
                              "target series, also classify some series as 'Other'.")
            time.sleep(3)
            gui.close()
            return

        # Prepare image tensors
        gui.update_status("2/6: Preparing image data (center slices)...")
        if gui.is_cancelled():
            logger.info("CNN training cancelled by user.")
            gui.close()
            return
        dicom_paths = master_df['Path']

        # Set up progress callback for preprocessing
        total_paths = len(dicom_paths)
        gui.set_determinate_mode(total_paths)

        def preprocessing_progress(current, total, message):
            """Callback to update GUI during preprocessing"""
            gui.update_progress(current)
            gui.update_status(f"2/6: {message}")

        # Try to load from cache first, or preprocess and cache. Worker count is
        # resolved from config.NUM_PARALLEL_WORKERS inside prepare_and_cache_cnn_data.
        # cancel_check lets the user abort preprocessing promptly.
        cache_path = prepare_and_cache_cnn_data(
            dicom_paths.tolist(),
            master_df['Classification'].tolist(),
            load_slices_func=processing_module.load_series_slices,
            output_dir=output_dir,
            progress_callback=preprocessing_progress,
            cancel_check=gui.is_cancelled,
        )

        gui.set_indeterminate_mode()

        if gui.is_cancelled():
            logger.info("CNN training cancelled by user.")
            gui.close()
            return

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
            model_files_dir = os.path.join(output_dir, "model_files")
            os.makedirs(model_files_dir, exist_ok=True)
            target_encoder_filename = os.path.join(model_files_dir, "label_encoder.joblib")
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
        if gui.is_cancelled():
            logger.info("CNN training cancelled by user.")
            gui.close()
            return

        # Set up progress callback for training
        gui.set_determinate_mode(int(_cfg('NUM_EPOCHS', NUM_EPOCHS)))

        def training_progress(current_epoch, total_epochs, loss, accuracy, message):
            """Callback to update GUI during training"""
            gui.update_progress(current_epoch)
            gui.update_status(f"4/6: {message}")

        # Train the model (this is a blocking call; caller should run in a thread)
        try:
            model_path, best_accuracy = train_cnn_model_logic(
                X, y_encoded, num_classes, output_dir,
                progress_callback=training_progress,
                cancel_check=gui.is_cancelled,
            )
        except Exception as e:
            gui.update_status(f"❌ Error during CNN training: {e}")
            time.sleep(2)
            gui.close()
            return

        gui.set_indeterminate_mode()

        # If cancelled, close immediately without saving summary
        if gui.is_cancelled():
            logger.info("CNN training cancelled by user — skipping model save summary.")
            gui.close()
            return

        # Save the target encoder
        try:
            joblib.dump(target_encoder, target_encoder_filename)
        except Exception as e:
            logger.warning(f"Failed to save target encoder: {e}")

        # Save a short summary
        try:
            final_message = f"✅ Image Model Training.\nModel Saved: {model_path}\nBest Accuracy: {best_accuracy:.2f}%"
            model_files_dir = os.path.join(output_dir, "model_files")
            summary_path = os.path.join(model_files_dir, "cnn_model_training_summary.txt")
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