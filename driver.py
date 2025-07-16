# driver.py

import matplotlib.pyplot as plt
import numpy as np
import time

from coclust.evaluation.external import accuracy # altered to use linear_sum_assignment instead of linear_assignment because of deprecation warning
from pyitlib import discrete_random_variable as drv
from sklearn import metrics

from depth_DBSCAN import *

def reorder(points, ground_truth, new_order, results):
    new_result = []
    size = len(points)

    for i in range(size):
        for j in range(size):
            if np.all(new_order[j] == points[i]):
                new_result.append(results[j])
                break

    return new_result

if len(sys.argv) < 5:
    print( 'Usage: exe [inputFile] [distance] [minimum points] [theta] [phi (only for m)]')
    exit()

# Read the input file
points = []
ground_truth = []
path = "Datasets/"
with open(path + sys.argv[1], 'r') as infile:
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
    distance = str(sys.argv[2]).lower()
    minPts = int(sys.argv[3])
    theta = float(sys.argv[4])
    phi = float(sys.argv[5])
except (ValueError, IndexError):
    print("Invalid Input: distance (mahalanobis/projection), minPts (int), theta (float), and phi (float)")
    print("Usage: exe [inputFile] [distance] [minimum points] [theta] [phi (only for m)]")
    exit()

# Start timing and running the clustering algorithm
start_time = time.time()

# Create covariance matrix
inmat = np.array(points)
covar, globalEigs, globalEigVecs =  computeCovarianceMatric(inmat)

# Runs the algorithm based off the chosen distance function
if distance in {"1", "mahalanobis", "m"}:
    MD = mahalanobisDepth()
    MD.inverseCovar = np.linalg.pinv(covar)

    results, lastLabel  = dbscanLaunch(inmat, theta, MD, minPts)

    results, new_order = localClusterReScan(inmat, globalEigs, globalEigVecs, theta, phi, results, MD, minPts, lastLabel)
    results = reorder(points, ground_truth, new_order, results)
    print("Phi:", phi)
elif distance in {"2", "projection", "p"}:
    # For proj depth, we get the eigenvector with max eigenval
    maxEigVec = globalEigVecs[:, -1]

    PD = projectionDepth()
    PD.compute1DMedianAbsoluteDeviation(inmat, maxEigVec)

    results, lastLabel = dbscanLaunch(inmat, theta, PD, minPts)
elif distance in {"3", "euclidean", "e"}:
    ED = euclideanDistance()
    theta = theta*-1 # For euclidean distance, theta is negative so we can use the same DBSCAN implementation
    results, lastLabel = dbscanLaunch(inmat, theta, ED, minPts)
elif distance in {"4", "both", "b"}:
    # For proj depth, we get the eigenvector with max eigenval
    # First do a Projection Depth DBSCAN
    maxEigVec = globalEigVecs[:, -1]

    PD = projectionDepth()
    PD.compute1DMedianAbsoluteDeviation(inmat, maxEigVec)

    results, lastLabel = dbscanLaunch(inmat, theta, PD, minPts)
    #now do a local cluster re-scan with MD

    MD = mahalanobisDepth()
    MD.inverseCovar = np.linalg.pinv(covar)

    results, new_order = localClusterReScan(inmat, globalEigs, globalEigVecs, theta, phi, results, MD, minPts, lastLabel)
    results = reorder(points, ground_truth, new_order, results)
    print("Phi:", phi)

else:
    print("Invalid Input: Distance Function does not exist. Use: mahalanobis, projection, or euclidean")
    print("Usage: exe [inputFile] [distance] [minimum points] [theta]")
    exit()

elapsed_time = time.time() - start_time

# Extracting clusters and noise from results
n_clusters = len(set(results)) - (1 if 0 in results else 0)
n_noise = list(results).count(0)

print("Theta:", theta)
print("Minimum Points:", minPts)
print("Estimated number of clusters:", n_clusters)
print("Estimated number of noise points:", n_noise)
#print("Accuracy: %0.3f" % accuracy(ground_truth, results))
#print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(ground_truth, results))
# VI requires both arrays to be type "int" instead of "numpy.int64", so the values are converted
#vi_ground_truth = [int(x) for x in ground_truth]
#print("Variation of Information: %0.3f" % drv.information_variation(ground_truth, results))
print("Time Elapsed: %0.3f seconds\n" % elapsed_time)



# Plot result
plot = np.c_[points, results]

# Seperates the noise from the data (this is assuming that class 0 is labeled as noise and no other class will be labeled 0)
noise = []
index = []
for i, row in enumerate(plot):
    if row[-1] == 0:
        noise.append(row)
        index.append(i)
plot = np.delete(plot, index, 0)
noise = np.array(noise)

if len(noise) > 0:
    plt.scatter(noise[:, 0], noise[:, 1], s=10, c='grey', linewidths=0.3)

plt.scatter(plot[:, 0], plot[:, 1], s=20, c=plot[:, -1], cmap='tab20', linewidths=0.3, edgecolors='k')
plt.show()
print( plot[:, -1])
