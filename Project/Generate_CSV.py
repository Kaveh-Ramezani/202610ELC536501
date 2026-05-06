import numpy as np
import pandas as pd
import os

print("Loading unified dataset...")
datasetPath = os.path.join("./DS", 'heg_unified_dataset.npz')
data = np.load(datasetPath)

# 1. Extract the training data
X_train = data['x_train']

# 2. Take the VERY FIRST trial and transpose it
# Flips from (Channels=200, Time=121) to (Time=121, Channels=200)
first_trial = X_train[0].T 

# 3. Convert to a Pandas DataFrame
# Dynamically generate 200 column names (CH1, CH2 ... CH200)
num_channels = first_trial.shape[1]
col_names = [f'CH{i}' for i in range(1, num_channels + 1)]
df = pd.DataFrame(first_trial, columns=col_names)

# 4. Insert the timestamp column
# Dynamically generate 121 timestamps (assuming 100ms gaps, Edge Impulse will adjust if your true Hz is different)
num_timesteps = first_trial.shape[0]
df.insert(0, 'timestamp', range(0, num_timesteps * 100, 100))

# 5. Save it
df.to_csv('calibration_sample.csv', index=False)
print(f"Success! Saved calibration_sample.csv with {num_timesteps} rows and {num_channels} channels.")