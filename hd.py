#HDBSCAN Implementation 

import sys
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# read the data file
if len(sys.argv) < 2:
    print( 'Usage: exe [inputFile] [minimum cluster size] ')
    exit()

# Read the input file
points = []
ground_truth = []

with open(sys.argv[1], 'r') as infile:
    infile.readline() # Skip the header

    for line in infile:
        if len(line) < 2:
            continue

        nums = line.replace("\n", "").split(",")
        nums = [float(i) for i in nums]

        points.append(tuple(nums[: -1]))
        #points.append((float(nums[0]), float(nums[1])))
        ground_truth.append(int(nums[-1]))

# Input Parameters
try:
    minPts = int(sys.argv[2])
except (ValueError, IndexError):
    print("Invalid Input: distance (mahalanobis/projection), minPts (int), theta (float), and phi (float)")
    print("Usage: exe [inputFile] [distance] [minimum points] [theta] [phi (only for m)]")
    exit()


# Initialize and fit the HDBSCAN model
# note if not using the Roadway data set,  you need to adjust the min_samples parameter (or jsut remove it)
clusterer = hdbscan.HDBSCAN(min_cluster_size=minPts, min_samples=3)
X = np.array( points)

clusterer.fit(X)

#  Get the cluster labels
# The 'labels_' attribute contains the assigned cluster for each data point.
# A label of -1 indicates a noise point (not assigned to any cluster).
labels = clusterer.labels_

# Plot the data points, colored by their assigned cluster label.
# Noise points (-1) can be colored differently for clarity.
plt.figure()
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [.5, .5, .5, 1]

    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)


plt.show()
