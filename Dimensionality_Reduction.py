import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('processed data/WFDBRecords/01/010/JS00001_denoised.csv')

# Extract features
X = data.iloc[:, 1:].values

# Determine the number of components dynamically
n_components = min(X.shape[0], X.shape[1])

# Perform PCA with dynamic number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Create dictionary for PCA results
pca_columns = {'time': data['time']}
for i in range(n_components):
    pca_columns[f'PC{i + 1}'] = X_pca[:, i]

# Create DataFrame for PCA results
df_pca = pd.DataFrame(pca_columns)

# Plot PCA components
plt.figure(figsize=(12, 6))
for i in range(n_components):
    plt.plot(df_pca['time'], df_pca[f'PC{i + 1}'], label=f'Principal Component {i + 1}')

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('ECG Signals Projected onto Principal Components')
plt.legend()
plt.show()
