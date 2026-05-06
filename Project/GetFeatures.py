from Misc import *
import subprocess
import sys
import os
import numpy as np

print_yellow("Loading unified dataset...")
datasetFolder = "./DS"
datasetPath = os.path.join(datasetFolder,'heg_unified_dataset.npz')

if os.path.exists(datasetFolder) and os.path.isfile(datasetPath):
  print_green("Dataset exists.")
else:
  print_red("Dataset Doesn't Exists.")
  print_yellow("Running Dataset_prepration.py")
  subprocess.run([sys.executable, "Dataset_prepration.py"])

data = np.load(datasetPath)

X_Train = data['x_train']

# --- THE FIX ---
# X_Train[0] is shape (200, 121). 
# We slice it to (200, 120) before flattening to get exactly 24,000 features
sample_features = X_Train[0, :, :120].flatten()
# ---------------

feature_string = ", ".join([f"{val:.6f}" for val in sample_features])

output_filename = "ei_test_features_24000.txt"
with open(output_filename, "w") as file:
    file.write(feature_string)

print_green(f"Success! Saved exactly {len(sample_features)} features to '{output_filename}'")