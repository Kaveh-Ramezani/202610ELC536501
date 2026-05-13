import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load the dataset
print("Loading dataset...")
datasetPath = os.path.join("./DS", 'heg_unified_dataset.npz')
data = np.load(datasetPath)

X_train = data['x_train']
y_train = data['y_train'] # Assuming your labels are saved as y_train

# 2. Define your class mapping
# NOTE: Change these numbers if your labels are mapped differently!
CLASS_REST = 0
CLASS_SQUEEZE = 1
CLASS_MOTION = 2

# Find the first index (trial) for each class
idx_rest = np.where(y_train == CLASS_REST)[0][1]
idx_squeeze = np.where(y_train == CLASS_SQUEEZE)[0][1]
idx_motion = np.where(y_train == CLASS_MOTION)[0][1]

# 3. Setup plotting parameters
# We will plot just one channel (e.g., Channel 0) so the graph is readable
channel_to_plot = 0 
num_timesteps = X_train.shape[2] # Should be 121

# Create a time axis in seconds (assuming 10Hz / 100ms per step)
time_axis = np.arange(0, num_timesteps) * 0.1

# 4. Generate the Plot
# Use a clean, academic style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle('HEG Signal Comparison Across Cognitive and Physical States', fontsize=16, fontweight='bold', y=0.95)

# --- Subplot 1: Rest ---
axs[0].plot(time_axis, X_train[idx_rest, channel_to_plot, :], color='#002c5f', linewidth=1.5)
axs[0].set_title('Rest State (Baseline Hemodynamics)', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Amplitude')

# --- Subplot 2: Squeeze (Motor Task) ---
axs[1].plot(time_axis, X_train[idx_squeeze, channel_to_plot, :], color='#22c55e', linewidth=1.5)
axs[1].set_title('Squeeze Task (Active Motor Cortex)', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Amplitude')

# --- Subplot 3: Motion Artifact ---
axs[2].plot(time_axis, X_train[idx_motion, channel_to_plot, :], color='#dc2626', linewidth=1.5)
axs[2].set_title('Motion Artifact (Environmental Noise)', fontsize=12, fontweight='bold')
axs[2].set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
axs[2].set_ylabel('Amplitude')

# Clean up layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.88) # Make room for the main title

# Save it as a high-res image for your PowerPoint
output_file = "Images/HEG_Signal_Comparison.svg"
plt.savefig(output_file, format='svg', bbox_inches='tight')
print(f"Success! High-resolution plot saved as {output_file}")

# Show the plot on screen
plt.show()