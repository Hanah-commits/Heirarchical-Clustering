import math
import numpy as np
import pandas as pd


input_df = pd.read_csv('methylation.csv', sep=';', na_values='.', decimal=',')
df = input_df.iloc[:, 7:]  # ignore irrelevant columns
df = df.fillna(0)

cell_types = df.columns
clusters = []
for x in range(len(cell_types)):  # initial cluster with each cel type as individual cluster
    clusters.append([])
    for y in range(1):
        clusters[x].append(cell_types[x])


def euclidean_distance(col_1, col_2):
    """
    distance metric for pairwise distances between methylation patterns
    of two cell types a and b
    """

    sqr_diff = (df[col_1] - df[col_2]).pow(2)
    sum_sqr_diff = sqr_diff.sum()
    return math.sqrt(sum_sqr_diff)


def clustering():
    """
    1. Compute a distance matrix (avg linkage criterion between all cell types)
    2. Find clusters with smallest distance between them
    3. Merge clusters
    4. Repeat steps 1-3 until only one cluster remains
    """

    while len(clusters) > 1:

        # STEP 1
        dm = [[] for i in range(len(clusters))]  # distance matrix
        cell_type1 = []  # holds elements of current cluster
        row = []  # holds computed values for each row of the distance matrix
        i = 0

        while i < len(clusters):

            cell_type1.extend(clusters[i])
            sum = 0
            for cell_type2 in clusters:

                if cell_type2 == cell_type1:
                    row.append(0.0)

                else:  # linkage criterion
                    for elem_i in cell_type1:
                        for elem_j in cell_type2:
                            sum += euclidean_distance(elem_i, elem_j)

                    linkage_val = sum / (len(cell_type1) * len(cell_type2))
                    row.append(linkage_val)
                    sum = 0

            dm[i] = row[:]
            row.clear()
            cell_type1.clear()
            i += 1

        # STEP 2
        dm = np.array(dm)
        smallest_linkage = np.min(dm[np.nonzero(dm)])
        smallest_dist = np.where(dm == smallest_linkage)

        # STEP 3
        if len(smallest_dist[0]) > 1:
            cell_1 = smallest_dist[0][0]
            cell_2 = smallest_dist[0][1]

            print('Merged clusters: ', clusters[cell_1] + clusters[cell_2])
            print('Linkage value  : ', smallest_linkage)

            clusters.append(clusters[cell_1] + clusters[cell_2])
            del clusters[cell_1]
            del clusters[cell_2 - 1]

        else:

            smallest_dist = np.array(smallest_dist)
            smallest_dist = smallest_dist.ravel()
            list(set(smallest_dist))
            smallest_dist.sort()
            cell_1 = smallest_dist[0]
            cell_2 = smallest_dist[1]

            print('Merged clusters: ', clusters[cell_1] + clusters[cell_2])
            print('Linkage value  : ', smallest_linkage)

            clusters.append(clusters[cell_1] + clusters[cell_2])
            del clusters[cell_1]
            del clusters[cell_2 - 1]

        print('Current cluster: ', clusters)


if __name__ == "__main__":
    clustering()
