import numpy as np
import os

print("Loading unified dataset...")
datasetPath = os.path.join("./DS", 'heg_unified_dataset.npz')
data = np.load(datasetPath)

# 1. Extract the augmented training data
X_train = data['x_train']

# 2. Take a small representative sample (e.g., the first 50 trials)
# Edge Impulse only needs a handful of samples to figure out the 8-bit math boundaries
calibration_data = X_train[:50]

# 3. Apply the exact same "Vertical 2D Trick" shape transformation 
# Changing shape from (50, 4, 120) to (50, 4, 120, 1)
calibration_data_4D = np.expand_dims(calibration_data, axis=3)

# 4. Save it explicitly as a .npy file
# Ensure the data type is float32 to match your ONNX model's expected input
output_filename = 'calibration_features.npy'
np.save(output_filename, calibration_data_4D.astype(np.float32))

print(f"Success! Saved {len(calibration_data_4D)} samples with shape {calibration_data_4D.shape} to {output_filename}")