import numpy as np
import matplotlib.pyplot as plt

# Dataset
original_times11 = [
    [760, 662, 797, 738, 764, 646],
    [760, 662, 658, 738, 764, 646],
    [760, 662, 658, 738, 658, 646],
    [694, 662, 658, 738, 658, 646],
    [694, 662, 658, 658, 658, 646],
    [658, 662, 658, 658, 658, 646],
    [658, 694, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [694, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [694, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [714, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [694, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646],
    [658, 658, 658, 658, 658, 646]
]

# Calculate average for each generation
averages = np.mean(original_times11, axis=1)

# Plot
plt.plot(range(1, 21), averages, marker='o')
plt.title('Average Processing Time per Generation')
plt.xlabel('Generation')
plt.ylabel('Average Processing Time (sec)')
plt.grid(True)
plt.xticks(np.arange(1, 21, step=1))
plt.show()
