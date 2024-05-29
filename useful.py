import csv
from biosppy.signals import ecg
import matplotlib.pyplot as plt

# Data points
ecg_signal = []

# Read ECG signal from the CSV file (assuming the signal is in the second column)
with open('processed data/WFDBRecords/01/010/JS00001.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the first row
    for row in reader:
        ecg_signal.append(float(row[8]))  # Lead V1 (heartbeat)

# Process the ECG signal and plot
ecg_results = ecg.ecg(signal=ecg_signal, sampling_rate=1000., show=True)

# Plot the ECG signal with R-peaks and templates
plt.figure(figsize=(12, 6))

