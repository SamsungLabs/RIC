import typing
import numpy as np
import open3d as o3d

def calculate_fscore(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud, th: float=0.01) -> typing.Tuple[float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall

def calculate_chamfer(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud):
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    #d1_squared = np.square(d1)
    chamfer_d1 = np.sum(d1) / len(d1)

    #d2_squared = np.square(d2)
    chamfer_d2 = np.sum(d2) / len(d2)

    return chamfer_d1 + chamfer_d2, chamfer_d1, chamfer_d2


def compute_iou(gt: o3d.geometry.PointCloud, pr: o3d.geometry.PointCloud, resolution: int=100):

    true_pts = np.asarray(gt.points)
    pred_pts = np.asarray(pr.points)


    grid_indices_true = ((true_pts + 0.5) * resolution).astype(int)
    grid_indices_pred = ((pred_pts + 0.5) * resolution).astype(int)

    grid_true = np.zeros((resolution, ) * 3)
    grid_pred = np.zeros((resolution, ) * 3)

    grid_true[grid_indices_true[:,0], grid_indices_true[:,1], grid_indices_true[:,2]] = 1
    grid_pred[grid_indices_pred[:,0], grid_indices_pred[:,1], grid_indices_pred[:,2]] = 1

    intersection = np.sum(np.minimum(grid_pred, grid_true))
    union = np.sum(np.maximum(grid_pred, grid_true))

    return intersection / union