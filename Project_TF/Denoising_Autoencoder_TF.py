import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# TensorFlow Imports
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Assuming Misc contains your print_green, print_yellow, etc.
from Misc import *

# Check the hardware
print(f"Is GPU available? {tf.config.list_physical_devices('GPU')}")

####################################################################################################
# Loading the dataset
####################################################################################################
# 1. Load the locked dataset
print_yellow("Loading unified dataset...")
datasetFolder = "./DS"
datasetPath = os.path.join(datasetFolder,'heg_unified_dataset_tf.npz')

if os.path.exists(datasetFolder) and os.path.isfile(datasetPath):
    print_green("Dataset exists.")
else:
    print_red("Dataset Doesn't Exists.")
    print_yellow("Running Dataset_prepration.py")
    subprocess.run([sys.executable, "Dataset_prepration_TF.py"])

data = np.load(datasetPath)

y_Raw = data['y_Raw']
X_Filtered = data['X_Filtered']

####################################################################################################
# Data Preparation
####################################################################################################
# Create the targets (Y) - The cleanest data we have
clean_indices = (y_Raw == 0) | (y_Raw == 1) # Only Rest and Squeeze
Y_clean_target = X_Filtered[clean_indices]

# Find the global standard deviation of your clean data
signal_scale = np.std(Y_clean_target)
# Divide everything by this scale to normalize the amplitude to ~1.0
Y_clean_target = Y_clean_target / signal_scale

# Create the Inputs (X) - Injecting severe synthetic noise
noise_level = 0.7 * np.std(Y_clean_target)
X_noisy_input = Y_clean_target + np.random.normal(0, noise_level, Y_clean_target.shape)

# Optional: Add simulated "Motion Spikes" to random channels to make it harder
spike_mask = np.random.choice([0, 1], size=Y_clean_target.shape, p=[0.98, 0.02])
X_noisy_input += spike_mask * 2.0

print(f"Original shape before processing: {X_noisy_input.shape}")

# 1. Precise Squeeze: Only remove the "Width" dimension (axis 2)
# This turns (N, 121, 1, 1) into (N, 121, 1)
if X_noisy_input.ndim == 4:
    X_noisy_input_tf = np.squeeze(X_noisy_input, axis=2).astype(np.float32)
    Y_clean_target_tf = np.squeeze(Y_clean_target, axis=2).astype(np.float32)
    print("Squeezed 4D -> 3D (Batch, Time, Channels)")
    
elif X_noisy_input.ndim == 3:
    # If it was already (Batch, Channels, Time), perform the flip
    X_noisy_input_tf = np.transpose(X_noisy_input, (0, 2, 1)).astype(np.float32)
    Y_clean_target_tf = np.transpose(Y_clean_target, (0, 2, 1)).astype(np.float32)
    print("Transposed 3D (Batch, Channels, Time) -> (Batch, Time, Channels)")
else:
    raise ValueError(f"CRITICAL ERROR: Unexpected shape {X_noisy_input.shape}")

num_timesteps = X_noisy_input_tf.shape[1]
num_channels = X_noisy_input_tf.shape[2] 

print(f"Final TF Shape for Autoencoder: {X_noisy_input_tf.shape}")
####################################################################################################
# Model Architecture
####################################################################################################
def build_autoencoder(timesteps, channels):
    # Input: (Batch, 121, Channels)
    inputs = layers.Input(shape=(timesteps, channels), name='rawMcuWave')
    
    # ENCODER
    x = layers.Conv1D(filters=16, kernel_size=11, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x) # 121 -> 60
    
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x) # 60 -> 30
    
    # DECODER
    x = layers.Conv1DTranspose(filters=16, kernel_size=2, strides=2, padding='valid', activation='relu')(x) # 30 -> 60
    
    # Final reconstruction layer
    x = layers.Conv1DTranspose(filters=channels, kernel_size=2, strides=2, padding='valid', activation='linear')(x) # 60 -> 120
    
    # CRITICAL FIX: Add 1 zero-padding at the end to match 121 samples
    decoded = layers.ZeroPadding1D(padding=(0, 1), name='cleanWave')(x) # 120 -> 121
    
    return models.Model(inputs=inputs, outputs=decoded)

autoencoder = build_autoencoder(num_timesteps, num_channels)
autoencoder.summary()

####################################################################################################
# Custom Loss & Training
####################################################################################################
# ShapeAwareLoss translated to TensorFlow
def shape_aware_loss(y_true, y_pred):
    alpha = 0.5
    # 1. Standard Point-wise Error
    loss_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 2. Derivative Error (Difference between adjacent time steps)
    # Slicing along the Time axis (axis 1 in TF)
    diff_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    loss_slope = tf.reduce_mean(tf.square(diff_true - diff_pred))
    
    # Combine them
    return loss_mse + (alpha * loss_slope)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.004)
autoencoder.compile(optimizer=optimizer, loss=shape_aware_loss)

# Learning Rate Scheduler (Cuts LR in half every 15 epochs)
def step_decay(epoch, lr):
    if epoch > 0 and epoch % 15 == 0:
        return lr * 0.5
    return lr
lr_scheduler = callbacks.LearningRateScheduler(step_decay)

epochs = 50
print_green("Training the AI Filter...")

# TensorFlow handles the batching, looping, and zeroing of gradients automatically
history = autoencoder.fit(
    x=X_noisy_input_tf,
    y=Y_clean_target_tf,
    batch_size=32,
    epochs=epochs,
    shuffle=True,
    callbacks=[lr_scheduler]
)

print_green("AI Filter Training Complete!")

####################################################################################################
# Evaluation & Plotting
####################################################################################################
# Grab a few samples for plotting
sample_noisy = X_noisy_input_tf[:4]
sample_clean = Y_clean_target_tf[:4]

# Run the noisy data through the AI Filter
reconstructed_wave = autoencoder.predict(sample_noisy)

# TF outputs are (Batch, Time, Channels)
num_plots = 2
channel_idx = 0

fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)

for i in range(num_plots):
    # Plot Noisy Input (Notice the slicing is [Batch, Time, Channel] instead of PyTorch's [Batch, Channel, Time])
    axes[i].plot(sample_noisy[i, :, channel_idx], 
                 label='Input: Noisy MCU Signal', 
                 color='lightgray', linewidth=1.5)

    # Plot Clean Target
    axes[i].plot(sample_clean[i, :, channel_idx], 
                 label='Target: Original Clean Wave', 
                 color='blue', linewidth=2, alpha=0.7)

    # Plot AI Output
    axes[i].plot(reconstructed_wave[i, :, channel_idx], 
                 label='Output: AI Reconstructed Wave', 
                 color='red', linestyle='dashed', linewidth=2)
    
    axes[i].set_ylabel(f"Sample {i+1}", fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.6)
    
    if i == 0:
        axes[i].legend(loc="upper right")

axes[-1].set_xlabel("Time Steps (10 Hz sampling)", fontsize=12)
fig.suptitle("Edge-HEG Denoising Autoencoder Performance", fontsize=16, y=0.98)

plt.tight_layout()
plt.savefig(f"Images/Denoising_Autoencoder_{num_plots}plots.svg")
# plt.show()

####################################################################################################
# 7. Exporting to TFLite (Fixed for Keras 3 / TF 2.16+)
####################################################################################################
print_green("\n--- 7. Exporting to TFLite ---")
os.makedirs("tf_exports", exist_ok=True)
autoencoder.save("tf_exports/edgeHeg_model_AE.keras")

float_path = "tf_exports/edge_heg_autoencoder_Float32.tflite"
int8_path = "tf_exports/edge_heg_autoencoder_INT8.tflite"

# --- THE WORKAROUND: Convert to Concrete Function ---
# This avoids the 'NoneType' error by creating a raw TF graph
run_model = tf.function(lambda x: autoencoder(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, num_timesteps, num_channels], tf.float32)
)

# --- FLOAT 32 EXPORT ---
converter_float = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model_float = converter_float.convert()
with open(float_path, "wb") as f:
    f.write(tflite_model_float)

# --- INT8 QUANTIZATION EXPORT ---
def representative_dataset_gen():
    for i in range(100):
        # Data must be (1, Time, Channels) to match concrete_func signature
        yield [X_noisy_input_tf[i:i+1].astype(np.float32)]

converter_quant = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_dataset_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.inference_input_type = tf.int8  
converter_quant.inference_output_type = tf.int8 

tflite_model_quant = converter_quant.convert()
with open(int8_path, "wb") as f:
    f.write(tflite_model_quant)

print_green("Success! TFLite models generated via Concrete Function.")

####################################################################################################
# 8. Hardware Deployment Summary (Flash & SRAM Profiling)
####################################################################################################
print_green("\n--- 8. Hardware Deployment Summary ---")

# 1. Calculate Flash Memory (Binary File Size)
if os.path.exists(float_path) and os.path.exists(int8_path):
    float_size = os.path.getsize(float_path) / 1024
    int8_size = os.path.getsize(int8_path) / 1024

    print_yellow(f"Float32 Model Size (Flash): {float_size:.2f} KB")
    print_yellow(f"INT8 Model Size (Flash):    {int8_size:.2f} KB")
    print_yellow(f"Memory Saved:               {(float_size - int8_size):.2f} KB ({(float_size/int8_size):.1f}x smaller)")

# 2. Estimate SRAM (RAM) Requirements for the INT8 Model
interpreter_summary = tf.lite.Interpreter(model_path=int8_path)
interpreter_summary.allocate_tensors()

tensor_details = interpreter_summary.get_tensor_details()
total_variable_memory = 0

for tensor in tensor_details:
    # 1 byte per INT8 value
    tensor_size_bytes = np.prod(tensor['shape']) * 1 
    
    # Exclude weights (Flash) to find actual RAM usage (activations)
    name = tensor['name'].lower()
    if 'weight' not in name and 'bias' not in name:
        total_variable_memory += tensor_size_bytes

# Add 2 KB overhead for the TFLite Micro framework
estimated_sram_kb = (total_variable_memory / 1024) + 2.0

print_yellow(f"\nEstimated Minimum SRAM (Tensor Arena): ~{estimated_sram_kb:.2f} KB")
print_yellow("*(When writing your C++ code, start your tensor_arena size at this number and adjust based on compiler feedback)*")

# 3. A Note on MACs
print_red("\n--- A Note on MACs (Multiply-Accumulates) ---")
print_red("The TFLite MLIR compiler calculated this during export!")
print_red("Look in the terminal logs above for 'Estimated count of arithmetic ops'.")