# DataDepthClustering

Code for clustering using DBSCAN with data depth statistics

## Overview

This repository contains code for experiments and images reported in the paper in the following paper:

`McKenney, M.; Tucek, D., 
Statistical Depth Measures in
Density-Based Clustering with
Automatic Adjustment for Skewed
Data. ISPRS Int. J. Geo-Inf. 2025` 

The paper explores density-based clustering using DBSCAN with statistical data depth measures, including Mahalanobis Distance.

## Usage

`st_louis_bridges.txt` contains coordinates of bridges in the St. Louis area.  This data set is used in the paper.

`driver.py` is the executable to run the DBSCAN algorithm with the various distance values.  `Depth_DBSCAN.py` contains the depth value calculations. Run `driver.py` with no arguments for usage instruction.  An example execution using Mahalanobis Distance on the St. Louis bridges data set is: 

`python driver.py st_louis_bridges.txt m 3 .98 .8'

`hd.py` performs HDBSCAN on a data file.  Run with no arguments for usage

`bridges.py` generates a map plotting the locations of bridges in the St. Louis Bridge data. It uses the stlouisbridges.csv file.

`dd.py` contains prototype code for DBSCAN with Mahalanobis Distance and Projection Distance.
