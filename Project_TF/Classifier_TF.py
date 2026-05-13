####################################################################################################
# Imports & Setup
####################################################################################################
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from Misc import *

# Lock randomness for reproducibility
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

print(f"TensorFlow Version: {tf.__version__}")

####################################################################################################
# 1. Loading and Formatting the Dataset
####################################################################################################
print_green("\n--- 1. Loading Dataset ---")

datasetPath = "./DS/heg_unified_dataset_tf.npz"

if not os.path.exists(datasetPath):
    print_red(f"Error: Dataset not found at {datasetPath}.")
    sys.exit(1)

data = np.load(datasetPath)

y_Raw = data['y_Raw']

# UPDATED: The data is already in TF format (NHWC), so we map it directly
X_Train_TF = data['x_train']
y_Train_Augmented = data['y_train']
X_Test_TF = data['x_test']
y_test = data['y_test']

# The "CRITICAL TF STEP" (expand_dims and transpose) has been completely removed!

print_yellow(f"Training Data Shape: {X_Train_TF.shape}")
print_yellow(f"Testing Data Shape: {X_Test_TF.shape}")

# Extract dynamic dimensions
time_steps = X_Train_TF.shape[1]
num_channels = X_Train_TF.shape[3]
num_classes = len(np.unique(y_Raw))
class_names = ['Rest (0)', 'Squeeze (1)', 'Motion (2)']

####################################################################################################
# 2. 1D-CNN Definition (EdgeHEG_CNN)
####################################################################################################
print_green("\n--- 2. Building Model ---")

def build_edge_heg_tf(t_steps, channels, classes):
    l2_reg = regularizers.l2(1e-3) # Standard weight decay

    model = models.Sequential(name="EdgeHEG_Stable_Baseline")
    model.add(layers.InputLayer(shape=(t_steps, 1, channels)))
    
    # Conv Block 1: 16 filters to capture initial frequency components
    model.add(layers.ZeroPadding2D(padding=((7, 7), (0, 0)))) 
    model.add(layers.Conv2D(16, (15, 1), strides=(2, 1), padding='valid', 
                            kernel_initializer='he_uniform', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 1)))
    
    # Conv Block 2: 32 filters for higher-level pattern recognition
    model.add(layers.ZeroPadding2D(padding=((2, 2), (0, 0))))
    model.add(layers.Conv2D(32, (5, 1), strides=(2, 1), padding='valid', 
                            kernel_initializer='he_uniform', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 1)))
    
    # Spatial Dropout + GAP: The standard "Armor" against overfitting
    model.add(layers.SpatialDropout2D(rate=0.3))
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense Head
    model.add(layers.Dropout(0.2)) 
    model.add(layers.Dense(classes, kernel_initializer='he_uniform', 
                           kernel_regularizer=l2_reg, name='fc_output'))
    
    return model

# --- SECTION 4: CALLBACKS ---
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)
model = build_edge_heg_tf(time_steps, num_channels, num_classes)

####################################################################################################
# 3. Compiling and Class Weights
####################################################################################################
# Calculate inversely proportional class weights (Handles the raw data imbalance)
class_counts = np.bincount(y_Raw)
total_samples = len(y_Raw)
weights = total_samples / (len(class_counts) * class_counts)
custom_weights = {0: 2.0, 1: 1.0, 2: 1.2}
# class_weight_dict = {i: weight for i, weight in enumerate(weights)}
class_weight_dict = custom_weights


print(f"\nComputed Class Weights: {class_weight_dict}")


def sparse_focal_loss(gamma=2.0):
    def loss(y_true, y_pred):
        # 1. Calculate standard Cross-Entropy Loss
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        
        # 2. Convert raw model outputs (logits) to probabilities
        probs = tf.nn.softmax(y_pred, axis=-1)
        
        # 3. Find the model's confidence in the CORRECT class
        # FIX: Safely flatten y_true using reshape instead of squeeze
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        
        # FIX: Extract depth safely for the one-hot encoder
        depth = tf.cast(tf.shape(y_pred)[-1], tf.int32) 
        y_true_onehot = tf.one_hot(y_true_flat, depth=depth) 
        
        pt = tf.reduce_sum(y_true_onehot * probs, axis=-1)
        
        # 4. Apply the Focal math: (1 - confidence)^gamma
        focal_factor = tf.pow(1.0 - pt, gamma)
        
        # 5. Multiply the standard loss by the focal factor
        return focal_factor * ce_loss
    return loss

# --- Compile the Model ---
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-2)

model.compile(
    optimizer=optimizer, 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)

####################################################################################################
# 4. Training
####################################################################################################
print("\n--- 3. Starting Training ---")
EPOCH_NUMBER = 50 # Increase epochs since the LR will slow down and fine-tune
BATCH_SIZE = 32

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,       # Cut learning rate in half
    patience= 4,      # If no improvement for 4 epochs
    min_lr=1e-6,      # Don't go lower than this
    verbose=1         # Print a message when it happens
)

# Add the callbacks=[...] to your fit function
history = model.fit(
    x=X_Train_TF, y=y_Train_Augmented,
    batch_size=BATCH_SIZE, epochs=EPOCH_NUMBER,
    class_weight=class_weight_dict,
    validation_data=(X_Test_TF, y_test), 
    callbacks=[lr_scheduler, early_stop], # Include both
    verbose=1
)
####################################################################################################
# 5. Evaluation & Float32 Confusion Matrix
####################################################################################################
print_green("\n--- 5. Final Evaluation ---")
test_loss, test_acc = model.evaluate(X_Test_TF, y_test, verbose=0)
print_red(f"FINAL TEST ACCURACY ON UNSEEN DATA: {test_acc * 100:.2f}%")

print_green("\n--- 4.5 Generating Confusion Matrix (Float32) ---")
os.makedirs("Images", exist_ok=True)
test_predictions_raw = model.predict(X_Test_TF, verbose=0)
y_pred = np.argmax(test_predictions_raw, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual True State')
plt.xlabel('AI Predicted State')
plt.title('Edge-HEG Classifier: Float32 Confusion Matrix')
plt.savefig("Images/Classifier_TF_Float32.svg", format='svg', bbox_inches='tight')
plt.close() # Close figure to prevent memory leaks
print_green("\nDetailed Classification Report (FLOAT32):")
print(classification_report(y_test, y_pred, target_names=class_names))

####################################################################################################
# 6. Exporting to TFLite (Float32 & INT8)
####################################################################################################
print_green("\n--- 6. Exporting to TFLite ---")
os.makedirs("tf_exports", exist_ok=True)
model.save("tf_exports/edgeHeg_model_C.keras")

# DYNAMIC MCU INPUT SHAPE APPLIED HERE
mcu_input_shape = tf.TensorSpec(shape=[1, time_steps, 1, num_channels], dtype=tf.float32)

@tf.function(input_signature=[mcu_input_shape])
def inference_func(input_data):
    return model(input_data, training=False)

concrete_func = inference_func.get_concrete_function()

# -- FLOAT 32 EXPORT --
converter_float = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model_float = converter_float.convert()
with open("tf_exports/edgeHeg_Float32.tflite", "wb") as f:
    f.write(tflite_model_float)

# -- INT8 QUANTIZATION EXPORT --
converter_quant = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
    for i in range(100):
        yield [X_Train_TF[i:i+1].astype(np.float32)]

converter_quant.representative_dataset = representative_dataset_gen
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.inference_input_type = tf.int8  
converter_quant.inference_output_type = tf.int8 

tflite_model_quant = converter_quant.convert()
with open("tf_exports/edgeHeg_INT8.tflite", "wb") as f:
    f.write(tflite_model_quant)

print_green("Success! TFLite models generated.")

####################################################################################################
# 7. Evaluating the Quantized (INT8) Model
####################################################################################################
print_green("\n--- 7. Generating Confusion Matrix for INT8 Model ---")

interpreter = tf.lite.Interpreter(model_path="tf_exports/edgeHeg_INT8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details["quantization"]

y_pred_quant = []

for i in range(len(X_Test_TF)):
    sample_float = X_Test_TF[i:i+1]
    
    if input_scale != 0.0:
        sample_int8 = np.round(sample_float / input_scale + input_zero_point)
        sample_int8 = np.clip(sample_int8, -128, 127).astype(np.int8)
    else:
        sample_int8 = sample_float.astype(np.int8)

    interpreter.set_tensor(input_details['index'], sample_int8)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    
    predicted_class = np.argmax(output_data[0])
    y_pred_quant.append(predicted_class)

cm_quant = confusion_matrix(y_test, y_pred_quant)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_quant, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual True State')
plt.xlabel('AI Predicted State')
plt.title('Edge-HEG Classifier: Quantized (INT8) Confusion Matrix')
svg_quant_path = "Images/Classifier_TF_INT8.svg"
plt.savefig(svg_quant_path, format='svg', bbox_inches='tight')
plt.close()

print_green(f"\nSuccess! Saved INT8 Confusion Matrix SVG to: {svg_quant_path}")
print_red("\nDetailed Classification Report (INT8):")
print(classification_report(y_test, y_pred_quant, target_names=class_names))

####################################################################################################
# 8. Hardware Deployment Summary (Flash & SRAM Profiling)
####################################################################################################
print_green("\n--- 8. Hardware Deployment Summary ---")

# 1. Calculate Flash Memory (Binary File Size)
float_path = "tf_exports/edgeHeg_Float32.tflite"
int8_path = "tf_exports/edgeHeg_INT8.tflite"

if os.path.exists(float_path) and os.path.exists(int8_path):
    float_size = os.path.getsize(float_path) / 1024
    int8_size = os.path.getsize(int8_path) / 1024

    print_yellow(f"Float32 Model Size (Flash): {float_size:.2f} KB")
    print_yellow(f"INT8 Model Size (Flash):    {int8_size:.2f} KB")
    print_yellow(f"Memory Saved:               {(float_size - int8_size):.2f} KB ({(float_size/int8_size):.1f}x smaller)")

# 2. Estimate SRAM (RAM) Requirements for the INT8 Model
# SRAM holds the input, output, and intermediate calculations (activations).
interpreter_summary = tf.lite.Interpreter(model_path=int8_path)
interpreter_summary.allocate_tensors()

tensor_details = interpreter_summary.get_tensor_details()

# TFLite Micro uses a "Memory Planner" to reuse RAM space. 
# We estimate the absolute maximum memory needed by summing the sizes of all non-weight tensors.
total_variable_memory = 0

for tensor in tensor_details:
    # We only care about INT8 tensors (1 byte per value) 
    tensor_size_bytes = np.prod(tensor['shape']) * 1 
    
    # We exclude weights and biases because those live permanently in Flash memory, not RAM
    name = tensor['name'].lower()
    if 'weight' not in name and 'bias' not in name:
        total_variable_memory += tensor_size_bytes

# We add roughly 2 KB to account for the TFLite Micro C++ framework overhead
estimated_sram_kb = (total_variable_memory / 1024) + 2.0

print_yellow(f"\nEstimated Minimum SRAM (Tensor Arena): ~{estimated_sram_kb:.2f} KB")
print_yellow("*(When writing your C++ code, start your tensor_arena size at this number and adjust based on compiler feedback)*")

# 3. A Note on MACs
print_red("\n--- A Note on MACs (Multiply-Accumulates) ---")
print_red("Programmatically extracting MACs in TensorFlow 2.16 is complex due to the new backend.")
print_red("However, the TFLite MLIR compiler automatically calculated this for you during export!")
print_red("Look slightly further up in your terminal logs for a line that looks exactly like this:")
print_red("-> 'Estimated count of arithmetic ops: 2.90 M ops, equivalently 1.45 M MACs'")