import random
import numpy as np

# Set the mean and standard deviation for the normal distribution
mean = 1050
std_dev = 0.08  # Small standard deviation to keep values within range
h=1900
l=600
a=200
# Generate values
for _ in range(10):
    # Randomly decide if we should introduce an outlier (values around 0.2)
    if random.random() < 0.2:  # 20% chance of an outlier
        value = np.random.uniform(a, l)  # Randomly select a value between 0.2 and 0.25
    else:
        value = np.random.normal(mean, std_dev)
    
    # Clip the value to ensure it lies between 0.4 and 0.5 (unless it's an outlier)
    if not (0.15 <= value <= 0.25):  # Do not clip outliers
        value = np.clip(value, l, h)

    # Print the value, formatted with 2 decimal places
    print(f"{value:.2f}", end=", ")
