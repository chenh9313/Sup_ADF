#! /bin/python

import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from multiprocessing import Pool

# Replace 'data.csv' with the actual file path if it's located in a different directory
file_path = 'finalSNPset_OnlyMaize.beagle_imputed.genotype0123_T.csv'

# Record the starting time
start_time = time.time()
# Read the file into a pandas DataFrame; 
# Read CSV file using 'C' engine: for larger files, the C engine can be significantly faster. 
df = pd.read_csv(file_path, index_col=0, sep=" ")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# Print the DataFrame
print(df.head())

# Assuming you have 'df' as the pandas DataFrame with your data

def run_kmeans(k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df.values)
    return kmeans.inertia_  # Return the sum of squared distances to the closest centroid (inertia)

if __name__ == '__main__':
    ks = range(1, 103)  # Values of K from 1 to 10
    num_cores = 5  # Number of CPU cores to use

    with Pool(processes=num_cores) as pool:
        results = pool.map(run_kmeans, ks)

    # Store the results in a pandas DataFrame
    result_df = pd.DataFrame({'K': ks, 'Inertia': results})
    print(result_df)
    # Write the result_df DataFrame to a CSV file
    result_df.to_csv('results_kmeans_parallel_findBestK.tsv', sep='\t', index=False)
