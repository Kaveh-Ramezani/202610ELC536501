####################################################################################################
# Imports
####################################################################################################
# PyTorch
import torch
# Hardware Check
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
# General
import numpy as np
# Read Dataset
import mne
import os
import openneuro
# Misc
from Misc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- LOCK RANDOMNESS FOR PERFECT REPRODUCIBILITY ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ---------------------------------------------------
####################################################################################################
# Function Definitions Compatible With Hardware
####################################################################################################
def MCU_Filter_EMAHighpass(raw_epochs, shift_bits=8):
  """
  Simulates the integer-based EMA DC-blocking filter of the MCU.
  Expects input shape: (Trials, Channels, Time)
  """
  alpha = 1.0 / (2 ** shift_bits) 
  ema_baseline = np.zeros_like(raw_epochs)
  ema_baseline[:, :, 0] = raw_epochs[:, :, 0] # Initialize
    
  for t in range(1, raw_epochs.shape[2]):
    ema_baseline[:, :, t] = ema_baseline[:, :, t-1] + alpha * (raw_epochs[:, :, t] - ema_baseline[:, :, t-1])
        
  ac_signal = raw_epochs - ema_baseline
  return ac_signal
####################################################################################################
# Download Dataset
####################################################################################################
datasetID = 'ds007420'
targetFolder = './heg_full_data'
markerFile = os.path.join(targetFolder, 'dataset_description.json')

# Check if the folder and the marker file exist
if os.path.exists(targetFolder) and os.path.isfile(markerFile):
  print_green(f"Dataset {datasetID} already exists in '{targetFolder}'. Skipping download.")
else:
  print_yellow(f"Dataset not found or incomplete. Starting download to '{targetFolder}'...")
  try:
    # The openneuro-py library will fetch the actual binary data
    openneuro.download(dataset=datasetID, target_dir=targetFolder)
    print("Download successfully completed!")
  except Exception as e:
    print(f"An error occurred during download: {e}")
####################################################################################################
# Preparing the Dataset
####################################################################################################
# Choosing first 10 subjects out of 12 for training
trainSubjects = ['170', '171', '173', '174', '176', '177', '179', '181', '182', '183'] # 10 out of 12
# Map sessions/tasks to specific class labels
# Class 0: Rest | Class 1: Ball Squeeze | Class 2: Motion Artifact
taskConfig = [
  {'ses': 'ses-01', 'task': 'task-Resting', 'label': 0, 'runs': [None]},
  {'ses': 'ses-02', 'task': 'task-BallSqueezing', 'label': 1, 'runs': ['run-01', 'run-02', 'run-03']},
  {'ses': 'ses-03', 'task': 'task-Motion', 'label': 2, 'runs': [None]}
]
X_TrainList = []
y_TrainList = []

print_yellow(f"Starting Edge-AI Data Extraction for {len(trainSubjects)} training subjects...")

for sub in trainSubjects:
  print(f"Processing Subject {sub}...")
  
  for config in taskConfig:
    for run in config['runs']:
      run_str = f"_{run}" if run else ""
      fileName = f"sub-{sub}_{config['ses']}_{config['task']}{run_str}_nirs.snirf"
      filePath = os.path.join(targetFolder, f"sub-{sub}", config['ses'], 'nirs', fileName)

      if not os.path.exists(filePath):
        print_red(f"  - Missing: {fileName}") # Don't remove this. This is for debugging
        continue
      
      try:
        # 1. Load Raw
        raw = mne.io.read_raw_snirf(filePath, preload=True, verbose=False)
        
        # 2. RESAMPLE TO 10 Hz - Make the shape uniform.
        raw.resample(10.0)
        
        # 3. Extract Events
        if config['label'] in [0, 2]: 
          events = mne.make_fixed_length_events(raw, id=config['label'], duration=15.0)
          event_dict = {'Segment': config['label']}
        else:
          events, event_dict = mne.events_from_annotations(raw, verbose=False)

        # 4. Epoching (12-second windows)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=12.0, 
                            baseline=None, preload=True, verbose=False)
        
        eData = epochs.get_data()
        
        # Ensure we only keep full 12-second windows (120+ samples)
        if eData.shape[2] < 120: 
          continue

        X_TrainList.append(eData)
        y_TrainList.append(np.full(len(eData), config['label']))
          
      except Exception as e:
        print_red(f"  - Error in {fileName}: {e}")

# 5. Final Concatenation
if X_TrainList:
  X_Raw = np.concatenate(X_TrainList, axis=0)
  y_Raw = np.concatenate(y_TrainList, axis=0) 
  
  print_yellow("\nBalancing the Training Dataset...")
  print(f"Original Class Distribution: {np.bincount(y_Raw)}")

  # --- THE BALANCING LOGIC ---
  # 1. Find the exact indices of each class
  idx_rest = np.where(y_Raw == 0)[0]
  idx_squeeze = np.where(y_Raw == 1)[0]
  idx_motion = np.where(y_Raw == 2)[0]

  # 2. Find the size of the smallest class (the bottleneck)
  min_samples = min(len(idx_rest), len(idx_squeeze), len(idx_motion))

  # 3. Randomly shuffle and truncate the majority classes to match the minority
  idx_rest_balanced = np.random.choice(idx_rest, min_samples, replace=False)
  idx_squeeze_balanced = np.random.choice(idx_squeeze, min_samples, replace=False)
  idx_motion_balanced = np.random.choice(idx_motion, min_samples, replace=False)

  # 4. Combine and shuffle the new balanced dataset
  balanced_indices = np.concatenate([idx_rest_balanced, idx_squeeze_balanced, idx_motion_balanced])
  np.random.shuffle(balanced_indices)

  # 5. Apply the balanced indices to the raw data
  X_Raw = X_Raw[balanced_indices]
  y_Raw = y_Raw[balanced_indices]
  # ---------------------------

  print(f"Balanced Class Distribution: {np.bincount(y_Raw)}")

  # Simulate MCU High-Pass Filter on the newly balanced data
  X_Filtered = MCU_Filter_EMAHighpass(X_Raw, shift_bits=8)
  
  print_green("\nExtraction and Balancing Successful!")
  print(f"Total Samples: {X_Filtered.shape[0]}")
  print(f"Signal Windows: {X_Filtered.shape[2]} time points per sample")
else:
  print_red("No data was loaded. Please check if the 'target_folder' path is correct.")

####################################################################################################
# Making The Dataset Bigger
####################################################################################################
print_green("Augmenting Dataset to force Shape-Aware Learning...")

# 1. Sensory Noise Augmentation
noiseLevel = 0.05 * np.std(X_Filtered)
X_Gaussian = X_Filtered + np.random.normal(0, noiseLevel, X_Filtered.shape)

# 2. NEW: Amplitude Scaling Augmentation
# Generate a random multiplier between 0.5 (weak signal) and 2.0 (strong signal) for EVERY window
np.random.seed(42)
random_scales = np.random.uniform(0.8, 1.2, size=(X_Filtered.shape[0], 1, 1))
X_Scaled = X_Filtered * random_scales

# 3. Concatenate everything to TRIPLE your dataset size
X_Train_Augmented = np.concatenate((X_Filtered, X_Gaussian, X_Scaled), axis=0)
y_Train_Augmented = np.concatenate((y_Raw, y_Raw, y_Raw), axis=0)

print_green(f"Original Balanced Training Samples: {X_Filtered.shape[0]}")
print_green(f"Final Augmented Training Samples: {X_Train_Augmented.shape[0]}")
####################################################################################################
# Test Dataset
####################################################################################################
testSubjects = ['184', '185'] # The subjects the model has NEVER seen
X_TestList, y_TestList = [], []

print("Extracting Test Data...")
for sub in testSubjects:
  for config in taskConfig:
    for run in config['runs']:
      run_str = f"_{run}" if run else ""
      fileName = f"sub-{sub}_{config['ses']}_{config['task']}{run_str}_nirs.snirf"
      filePath = os.path.join(targetFolder, f"sub-{sub}", config['ses'], 'nirs', fileName)

      if not os.path.exists(filePath):
        print_red(f"  - Missing: {fileName}") # Don't remove this. This is for debugging
        continue
      
      try:
        raw = mne.io.read_raw_snirf(filePath, preload=True, verbose=False)
        raw.resample(10.0) # Match training frequency
        
        if config['label'] in [0, 2]: 
          events = mne.make_fixed_length_events(raw, id=config['label'], duration=15.0)
          event_dict = {'Segment': config['label']}
        else:
          events, event_dict = mne.events_from_annotations(raw, verbose=False)

        epochs = mne.Epochs(raw, events, tmin=0, tmax=12.0, baseline=None, preload=True, verbose=False)
        e_data = epochs.get_data()
        
        if e_data.shape[2] < 120: continue

        X_TestList.append(e_data)
        y_TestList.append(np.full(len(e_data), config['label']))
      except Exception as e:
        print_red(f"  - Error in {fileName}: {e}")

if not X_TestList:
  print_red("No data was loaded. Please check if the 'target_folder' path is correct.")
  raise("Error: Could not load test subjects.")

# Concatenate and apply the MCU filter
X_test_raw = np.concatenate(X_TestList, axis=0)
y_test = np.concatenate(y_TestList, axis=0)
X_test_filtered = MCU_Filter_EMAHighpass(X_test_raw, shift_bits=8)

# Convert to PyTorch Tensors
X_TestTensor = torch.tensor(X_test_filtered, dtype=torch.float32).to(device)
y_TestTensor = torch.tensor(y_test, dtype=torch.long).to(device)

####################################################################################################
# Saving
####################################################################################################
print_yellow("Saving unified dataset to disk...")
np.savez_compressed('DS/heg_unified_dataset.npz', 
                    y_Raw = y_Raw,
                    X_Raw = X_Raw,
                    X_Filtered = X_Filtered,
                    x_train=X_Train_Augmented, 
                    y_train=y_Train_Augmented, 
                    x_test=X_TestTensor.cpu().numpy(), # Save as numpy
                    y_test=y_TestTensor.cpu().numpy())
print_green("Dataset locked and saved!")
