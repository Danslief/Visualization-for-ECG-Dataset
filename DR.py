import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load ECG data from the CSV file
data = pd.read_csv('processed data/WFDBRecords/01/010/JS00001.csv')

# Extract ECG signal data (excluding the 'time' column)
ecg_signals = data.iloc[:, 1:].values

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
pca_result = pca.fit_transform(ecg_signals)

# Create a new DataFrame with the principal components and time
df_pca = pd.DataFrame({'time': data['time'], 'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1]})

# Plot the principal components against time
plt.figure(figsize=(12, 6))
plt.plot(df_pca['time'], df_pca['PC1'], label='PC1')
plt.plot(df_pca['time'], df_pca['PC2'], label='PC2')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('ECG Signals in Two Principal Components')
plt.legend()
plt.show()
