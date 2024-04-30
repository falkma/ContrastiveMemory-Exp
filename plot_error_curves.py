import numpy as np
import matplotlib.pyplot as plt

# Load error metrics from training log files
training_curves = {}
training_curves['training error'] = np.load('train_error_convolution.npy')
training_curves['validation error'] = np.load('val_error_convolution.npy')

# Set window size for averaging errors, equivalent to the number of batches per epoch
ew = 500  # window size for training error averaging
vw = 100  # window size for validation error averaging

# Calculate the averaged training and validation errors over the specified windows
training_curves['training error downsample'] = [
    np.mean(training_curves['training error'][i*ew:(i+1)*ew]) 
    for i in range(len(training_curves['training error']) // ew)
]
training_curves['validation error downsample'] = [
    np.mean(training_curves['validation error'][i*vw:(i+1)*vw]) 
    for i in range(len(training_curves['validation error']) // vw)
]

# Plot training and validation error curves on a logarithmic scale
plt.figure(figsize=(4,3))
plt.plot(np.log10(np.array(training_curves['training error downsample'])/100), label='train', color='grey', linewidth=2)
plt.plot(np.log10(np.array(training_curves['validation error downsample'])/100), label='test', color='black', linewidth=2, linestyle='--')

# Set labels for axes and legend
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.legend()

# Save the figure as a PNG and a PDF with tight bounding box
plt.savefig('training_curves_convolution.png', bbox_inches="tight")
plt.savefig('training_curves_convolution.pdf', bbox_inches="tight")

# Display the plot
plt.show()

# Save the downsampled error data for further analysis
np.save('train_error_vs_epoch_3B.npy', np.array(training_curves['training error downsample']))
np.save('test_error_vs_epoch_3B.npy', np.array(training_curves['validation error downsample']))
