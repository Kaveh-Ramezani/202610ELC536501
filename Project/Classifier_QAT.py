####################################################################################################
# Imports
####################################################################################################
# PyTorch
import torch.nn as nn
import torch.nn.functional as nnFunc
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
# Hardware Check
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
# General
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# Misc
from Misc import *
import subprocess
import sys
import os

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
EPOCH_NUMBER = 20
####################################################################################################
# Loading the dataset
####################################################################################################
# 1. Load the locked dataset
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

y_Raw = data['y_Raw']
X_Raw = data['X_Raw']
X_Filtered = data['X_Filtered']
X_Train_Augmented = data['x_train']
y_Train_Augmented = data['y_train']
X_test_filtered = data['x_test']
y_test = data['y_test']

# 2. Convert to PyTorch Tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_TestTensor = torch.tensor(X_test_filtered, dtype=torch.float32).to(device)
y_TestTensor = torch.tensor(y_test, dtype=torch.long).to(device)

####################################################################################################
# Custom QAT Components & 1D-CNN Definition
####################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as nnFunc

####################################################################################################
# 1. The Straight-Through Estimator (STE)
####################################################################################################
class FakeQuantizeSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, weight, scale, q_max):
    q_weight = torch.round(weight / scale)
    q_weight = torch.clamp(q_weight, -q_max, q_max)
    return q_weight * scale

  @staticmethod
  def backward(ctx, grad_output):
  # Prevent gradients from dying on the integer staircase
    return grad_output, None, None

####################################################################################################
# Custom QAT Layers (With ONNX-safe clamping)
####################################################################################################
class QAT_Conv1d(nn.Conv1d):
  def __init__(self, *args, bit_width=8, **kwargs):
    super().__init__(*args, **kwargs)
    self.bit_width = bit_width

  def forward(self, input):
    if self.bit_width >= 32:
      return super().forward(input)
        
    q_max = (2 ** (self.bit_width - 1)) - 1
    max_val = torch.max(torch.abs(self.weight))
    
    # Static ONNX-safe math to prevent division by zero
    scale = torch.clamp(max_val, min=1e-8) / q_max
    
    quantized_weight = FakeQuantizeSTE.apply(self.weight, scale, q_max)
    
    return nnFunc.conv1d(input, quantized_weight, self.bias, self.stride, 
                          self.padding, self.dilation, self.groups)

class QAT_Linear(nn.Linear):
  def __init__(self, *args, bit_width=8, **kwargs):
    super().__init__(*args, **kwargs)
    self.bit_width = bit_width

  def forward(self, input):
    if self.bit_width >= 32:
      return super().forward(input)
        
    q_max = (2 ** (self.bit_width - 1)) - 1
    max_val = torch.max(torch.abs(self.weight))
    
    # Static ONNX-safe math
    scale = torch.clamp(max_val, min=1e-8) / q_max
    
    quantized_weight = FakeQuantizeSTE.apply(self.weight, scale, q_max)
    
    return nnFunc.linear(input, quantized_weight, self.bias)

####################################################################################################
# The Optimized QAT Model Architecture
####################################################################################################
class EdgeHEG_CNN_QAT(nn.Module):
    def __init__(self, num_channels, num_classes, target_bit_width=8):
        super(EdgeHEG_CNN_QAT, self).__init__()
        self.target_bit_width = target_bit_width
        
        # Block 1: WIDE but SHALLOW (8 channels)
        self.conv1 = QAT_Conv1d(in_channels=num_channels, out_channels=8, kernel_size=15, stride=2, padding=7, bit_width=target_bit_width)
        self.bn1 = nn.BatchNorm1d(8) # Preserves amplitude differences between Rest and Squeeze
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2: (16 channels)
        self.conv2 = QAT_Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2, bit_width=target_bit_width)
        self.bn2 = nn.BatchNorm1d(16) 
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Aggressive Dropout to prevent memorization
        self.dropout = nn.Dropout(p=0.5) 
        self.flatten = nn.Flatten()
        
        # 16 channels * 7 remaining time steps = 112 features
        self.fc = QAT_Linear(16 * 7, num_classes, bit_width=target_bit_width)
          
    def forward(self, x):
        x = self.pool1(nnFunc.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nnFunc.relu(self.bn2(self.conv2(x))))
        
        x = self.dropout(x)
        x = self.flatten(x) 
        x = self.fc(x)
        return x
####################################################################################################
# Instantiate The Model
####################################################################################################
class HEGDataset(Dataset):
  def __init__(self, X, y):
    # Convert NumPy arrays to PyTorch Tensors
    self.X = torch.tensor(X, dtype=torch.float32)
    # Classification targets in PyTorch MUST be Long tensors
    self.y = torch.tensor(y, dtype=torch.long) 
        
  def __len__(self):
    return len(self.X)
    
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

# Instantiate DataLoader (Batching the data for memory efficiency)
trainDataset = HEGDataset(X_Train_Augmented, y_Train_Augmented)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)

numChannels = X_Filtered.shape[1]
numClasses = len(np.unique(y_Raw))

# --- SET YOUR TARGET HARDWARE BIT-WIDTH HERE ---
# Options: 8, 16, 32, 64
HARDWARE_BIT_WIDTH = int(input("Hardware Bit Width [8, 16, 32, 64]:"))
# -----------------------------------------------

# Instantiate the CNN with the QAT constraint
model = EdgeHEG_CNN_QAT(numChannels, numClasses, target_bit_width=HARDWARE_BIT_WIDTH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print_green(f"Model Initialized for {HARDWARE_BIT_WIDTH}-bit QAT on {device}!")
####################################################################################################
# Training
####################################################################################################
# Count how many samples of each class we actually have in the training set
class_counts = np.bincount(y_Raw)
total_samples = len(y_Raw)

# Calculate inversely proportional weights 
# Formula: Total Samples / (Number of Classes * Samples in this Class)
# Making rare classes heavily weighted and common classes lightly weighted
weights = total_samples / (len(class_counts) * class_counts)

# Convert to a PyTorch tensor and move to GPU
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

print(f"Sample Counts per class: {class_counts}")
print(f"Computed Penalties (Weights): {weights}")

# Give the weights to Loss Function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training and optimizer characteristics
epochs = EPOCH_NUMBER
learning_rate = 0.001
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

print_yellow(f"Starting Training on {device} for {epochs} Epochs...")

# Track loss and accuracy to see how the model improves
lossHistory = []
accHistory = []

for epoch in range(epochs):
  model.train() # Set model to training mode
  
  runningLoss = 0.0
  correctPredictions = 0
  totalSamples = 0
  
  # Iterate over the batches from your DataLoader
  for inputs, labels in trainLoader:
    # Move data to GPU if available
    inputs, labels = inputs.to(device), labels.to(device)
    
    # 1. Zero the gradients from the previous step
    optimizer.zero_grad()
    
    # 2. Forward Pass: Get the AI's guesses
    outputs = model(inputs)
    
    # 3. Calculate how wrong the guesses are
    loss = criterion(outputs, labels)
    
    # 4. Backward Pass: Calculate gradients
    loss.backward()
    
    # 5. Optimize: Update the CNN weights
    optimizer.step()
    
    # --- Tracking Stats ---
    runningLoss += loss.item() * inputs.size(0)
    
    # Find the class with the highest probability
    _, predicted = torch.max(outputs.data, 1)
    totalSamples += labels.size(0)
    correctPredictions += (predicted == labels).sum().item()
      
  # Calculate average loss and accuracy for this epoch
  epoch_loss = runningLoss / totalSamples
  epoch_acc = (correctPredictions / totalSamples) * 100
  
  lossHistory.append(epoch_loss)
  accHistory.append(epoch_acc)
  
  # Print progress every 5 epochs
  if (epoch + 1) % 5 == 0 or epoch == 0:
    print_yellow(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

print_green("\nTraining Complete!")

# 3. SAVE THE MODEL FOR HARDWARE
# Save the trained weights so they can be exported to C/C++ later
torch.save(model.state_dict(), f'onnx/edge_heg_classifier_INT{HARDWARE_BIT_WIDTH}.pth')
print_green(f"Model weights saved to 'edge_heg_classifier_INT{HARDWARE_BIT_WIDTH}.pth'")
####################################################################################################
# Testing
####################################################################################################

# 2. EVALUATING THE MODEL
model.eval() # Tell PyTorch we are evaluating, not training
with torch.no_grad(): # Save memory by turning off gradient tracking
  test_outputs = model(X_TestTensor)
  _, test_predictions = torch.max(test_outputs, 1)
    
# Calculate Final Accuracy
correct = (test_predictions == y_TestTensor).sum().item()
test_acc = (correct / len(y_TestTensor)) * 100
print_green(f"\nFINAL TEST ACCURACY ON UNSEEN DATA: {test_acc:.2f}%")

# GENERATING CONFUSION MATRIX
# Move tensors back to CPU for Scikit-Learn
y_true = y_TestTensor.cpu().numpy()
y_pred = test_predictions.cpu().numpy()

cm = confusion_matrix(y_true, y_pred)
class_names = ['Rest (0)', 'Squeeze (1)', 'Motion (2)']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual True State')
plt.xlabel('AI Predicted State')
plt.title('Edge-HEG Classifier: Test Set Confusion Matrix')
plt.savefig(f"./Images/Classifier_QAT_INT{HARDWARE_BIT_WIDTH}.svg")
# plt.show()

print_green("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
####################################################################################################
# Extracting The Model
####################################################################################################
# Move the model off the GPU and onto the CPU
model.cpu()
# Lock the model into evaluation mode (This triggers our 'else' block bypass!)
model.eval()

# Shape: (Batch=1, Channels=numChannels, TimeSteps=120)
dummyMcuInput = torch.randn(1, numChannels, 120)

# FORCE CREATE DIRECTORY IF MISSING
os.makedirs("onnx", exist_ok=True)
onnxFilePath = f"onnx/edgeHegClassifier_QAT_INT{HARDWARE_BIT_WIDTH}.onnx"

print_green("Tracing the computation graph...")
torch.onnx.export(
  model,                            
  dummyMcuInput,                    
  onnxFilePath,                     
  export_params=True,               
  opset_version=18,                 
  do_constant_folding=True,         
  input_names=['filteredADCData'],  
  output_names=['Class'],           
)
print_green(f"ONNX Model successfully saved to {onnxFilePath}!")