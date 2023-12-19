import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d

def filter_pointclouds(pointclouds):
    # Takes in a list of o3d point clouds and only keeps points that 3 of the point clouds predict
    all_added = []
    all_added_pcd = o3d.geometry.PointCloud()
    for i in range(len(pointclouds)):
        # rest_of_pcds = o3d.o3d.geometry.PointCloud()
        num_close = np.zeros(len(np.asarray(pointclouds[i].points)))
        for j in range(0, len(pointclouds)):
            if j == i:
                continue
            dists = pointclouds[i].compute_point_cloud_distance(pointclouds[j])
            dists = np.asarray(dists)
            ind = np.where(dists < 0.01)[0]

            hits = np.zeros(len(num_close))
            hits[ind] = 1

            num_close += hits

        ind_all = np.asarray(num_close >= 3).nonzero()[0]
        filtered = pointclouds[i].select_by_index(ind_all)

        all_added_pcd += filtered
        all_added.append(filtered)

    return all_added_pcd