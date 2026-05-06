####################################################################################################
# Imports & Setup
####################################################################################################
import tensorflow as tf
import numpy as np
import mne
import os
import openneuro
from Misc import *

# --- HARDWARE CHECK ---
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs Available: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"  - {gpu.name}")

# --- LOCK RANDOMNESS FOR PERFECT REPRODUCIBILITY ---
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
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

if os.path.exists(targetFolder) and os.path.isfile(markerFile):
    print_green(f"Dataset {datasetID} already exists in '{targetFolder}'. Skipping download.")
else:
    print_yellow(f"Dataset not found or incomplete. Starting download to '{targetFolder}'...")
    try:
        openneuro.download(dataset=datasetID, target_dir=targetFolder)
        print("Download successfully completed!")
    except Exception as e:
        print(f"An error occurred during download: {e}")

####################################################################################################
# Preparing the Dataset
####################################################################################################
trainSubjects = ['170', '171', '173', '174', '176', '177', '179', '181', '182', '183'] 

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
                print_red(f"  - Missing: {fileName}") 
                continue
            
            try:
                raw = mne.io.read_raw_snirf(filePath, preload=True, verbose=False)
                raw.resample(10.0)
                
                if config['label'] in [0, 2]: 
                    events = mne.make_fixed_length_events(raw, id=config['label'], duration=15.0)
                else:
                    events, _ = mne.events_from_annotations(raw, verbose=False)

                epochs = mne.Epochs(raw, events, tmin=0, tmax=12.0, baseline=None, preload=True, verbose=False)
                eData = epochs.get_data()
                
                if eData.shape[2] < 120: 
                    continue

                X_TrainList.append(eData)
                y_TrainList.append(np.full(len(eData), config['label']))
                
            except Exception as e:
                print_red(f"  - Error in {fileName}: {e}")

if X_TrainList:
    X_Raw = np.concatenate(X_TrainList, axis=0)
    y_Raw = np.concatenate(y_TrainList, axis=0) 
    
    print_yellow("\nBalancing the Training Dataset...")
    
    idx_rest = np.where(y_Raw == 0)[0]
    idx_squeeze = np.where(y_Raw == 1)[0]
    idx_motion = np.where(y_Raw == 2)[0]

    min_samples = min(len(idx_rest), len(idx_squeeze), len(idx_motion))

    idx_rest_balanced = np.random.choice(idx_rest, min_samples, replace=False)
    idx_squeeze_balanced = np.random.choice(idx_squeeze, min_samples, replace=False)
    idx_motion_balanced = np.random.choice(idx_motion, min_samples, replace=False)

    balanced_indices = np.concatenate([idx_rest_balanced, idx_squeeze_balanced, idx_motion_balanced])
    np.random.shuffle(balanced_indices)

    X_Raw = X_Raw[balanced_indices]
    y_Raw = y_Raw[balanced_indices]

    X_Filtered = MCU_Filter_EMAHighpass(X_Raw, shift_bits=8)
    
    print_green("Extraction and Balancing Successful!")
else:
    print_red("No data was loaded. Please check paths.")
    sys.exit(1)

####################################################################################################
# Making The Dataset Bigger
####################################################################################################
print_green("Augmenting Dataset...")

noiseLevel = 0.05 * np.std(X_Filtered)
X_Gaussian = X_Filtered + np.random.normal(0, noiseLevel, X_Filtered.shape)

np.random.seed(42)
random_scales = np.random.uniform(0.8, 1.2, size=(X_Filtered.shape[0], 1, 1))
X_Scaled = X_Filtered * random_scales

X_Train_Augmented = np.concatenate((X_Filtered, X_Gaussian, X_Scaled), axis=0)
y_Train_Augmented = np.concatenate((y_Raw, y_Raw, y_Raw), axis=0)

####################################################################################################
# Test Dataset
####################################################################################################
testSubjects = ['184', '185'] 
X_TestList, y_TestList = [], []

print("Extracting Test Data...")
for sub in testSubjects:
    for config in taskConfig:
        for run in config['runs']:
            run_str = f"_{run}" if run else ""
            fileName = f"sub-{sub}_{config['ses']}_{config['task']}{run_str}_nirs.snirf"
            filePath = os.path.join(targetFolder, f"sub-{sub}", config['ses'], 'nirs', fileName)

            if not os.path.exists(filePath):
                continue
            try:
                raw = mne.io.read_raw_snirf(filePath, preload=True, verbose=False)
                raw.resample(10.0) 
                
                if config['label'] in [0, 2]: 
                    events = mne.make_fixed_length_events(raw, id=config['label'], duration=15.0)
                else:
                    events, _ = mne.events_from_annotations(raw, verbose=False)

                epochs = mne.Epochs(raw, events, tmin=0, tmax=12.0, baseline=None, preload=True, verbose=False)
                e_data = epochs.get_data()
                
                if e_data.shape[2] < 120: continue

                X_TestList.append(e_data)
                y_TestList.append(np.full(len(e_data), config['label']))
            except Exception as e:
                pass

X_test_raw = np.concatenate(X_TestList, axis=0)
y_test = np.concatenate(y_TestList, axis=0)
X_test_filtered = MCU_Filter_EMAHighpass(X_test_raw, shift_bits=8)

####################################################################################################
# TensorFlow Formatting & Saving
####################################################################################################
print_yellow("Formatting arrays to NHWC for TensorFlow...")

def format_for_tf(data_array):
    # Expand dims to add "Width=1" -> Shape becomes (Trials, Channels, Time, 1)
    data_4d = np.expand_dims(data_array, axis=3)
    # Transpose to NHWC -> Shape becomes (Trials, Time, 1, Channels)
    return np.transpose(data_4d, (0, 2, 3, 1))

X_Raw_TF = format_for_tf(X_Raw)
X_Filtered_TF = format_for_tf(X_Filtered)
X_Train_Augmented_TF = format_for_tf(X_Train_Augmented)
X_test_filtered_TF = format_for_tf(X_test_filtered)

print_yellow("Saving unified dataset to disk...")
os.makedirs('DS', exist_ok=True)
np.savez_compressed('DS/heg_unified_dataset_tf.npz', 
                    y_Raw = y_Raw,
                    X_Raw = X_Raw_TF,
                    X_Filtered = X_Filtered_TF,
                    x_train = X_Train_Augmented_TF, 
                    y_train = y_Train_Augmented, 
                    x_test = X_test_filtered_TF, 
                    y_test = y_test)

print_green("Dataset formatted for TensorFlow and saved!")