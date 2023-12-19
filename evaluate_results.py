import typing
import numpy as np
import open3d as o3d
import cv2
import os
import copy
from scipy.spatial.transform import Rotation as R

from utils.iterative_utils import convert_realsense_rgb_depth_to_o3d_pcl

from utils.evaluate_results_utils import calculate_chamfer, calculate_fscore, compute_iou

import argparse

def normalize_pc(gt_points, pred_points):
    centroid = np.mean(gt_points, axis=0)
    gt_points -= centroid
    pred_points -= centroid
    furthest_distance = (np.max(np.sqrt(np.sum(abs(gt_points)**2,axis=-1))) * 2)
    gt_points /= furthest_distance
    pred_points /= furthest_distance
    return gt_points, pred_points

def evaluate_results(results_dir, gt_dir):
    pred_scenes = os.listdir(results_dir)
    total_fscore_list = []
    total_cscore_list = []
    total_f_cscore_list = []
    total_b_cscore_list = []
    total_iscore_list = []
    for pred_scene in pred_scenes:
        pred_frames = os.listdir(results_dir + pred_scene)
        for pred_frame in pred_frames:
            gt_pcd = o3d.io.read_point_cloud(gt_dir + pred_scene + "_gt.ply")
            pred_pcd = o3d.io.read_point_cloud(results_dir + pred_scene + "/" + pred_frame + "/final.ply")

            c1_T_w = np.load(results_dir + pred_scene + "/" + pred_frame + "/extrinsics.npy")

            pred_pcd = copy.deepcopy(pred_pcd).transform(c1_T_w)
            pred_pcd.scale(100, center=[0,0,0])
            c1_T_w[:3,-1] *= 100
            pred_pcd = copy.deepcopy(pred_pcd).transform(np.linalg.inv(c1_T_w))
            pred_pcd = pred_pcd.voxel_down_sample(voxel_size=0.5)
            pred_pcd.paint_uniform_color([1, 0, 0])

            z_min = np.min(np.array(gt_pcd.points), 0)[2]
        
            min_bound_meshes = np.load(gt_dir + pred_scene + "_min_bound.npy")
            max_bound_meshes = np.load(gt_dir + pred_scene + "_max_bound.npy")
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound_meshes, max_bound=max_bound_meshes)
            pred_pcd = pred_pcd.crop(bbox)

            # o3d.visualization.draw([gt_pcd] + [pred_pcd])

            pred_points = pred_pcd.points 
            gt_points = gt_pcd.points
            gt_points_normalize, pred_points_normalize = normalize_pc(gt_points, pred_points)


            pred_pcd.points = o3d.utility.Vector3dVector(pred_points_normalize)
            gt_pcd.points = o3d.utility.Vector3dVector(gt_points_normalize)

            
            c_score, c_forward, c_backward = calculate_chamfer(gt_pcd, pred_pcd)
            f_score, _, _ = calculate_fscore(gt_pcd, pred_pcd, 0.01)

            pred_pcd = pred_pcd.crop(gt_pcd.get_axis_aligned_bounding_box())
            i_score = compute_iou(gt_pcd, pred_pcd, 100)

            total_cscore_list.append(c_score)
            total_f_cscore_list.append(c_forward)
            total_b_cscore_list.append(c_backward)
            total_fscore_list.append(f_score)
            total_iscore_list.append(i_score)
    avg_fscore = np.mean(total_fscore_list)
    avg_cscore = np.mean(total_cscore_list)
    avg_f_cscore = np.mean(total_f_cscore_list)
    avg_b_cscore = np.mean(total_b_cscore_list)
    avg_iscore = np.mean(total_iscore_list)

    std_fscore = np.std(total_fscore_list)
    std_cscore = np.std(total_cscore_list)
    std_f_cscore = np.std(total_f_cscore_list)
    std_b_cscore = np.std(total_b_cscore_list)
    std_iscore = np.std(total_iscore_list)

    print("OUR METHOD:")
    print("F-Score Mean: ", np.around(avg_fscore, decimals=3), "F-Score STD: ", np.around(std_fscore, decimals=3))
    print("Chamfer Distance Mean: ", np.around(avg_cscore, decimals=5), "Chamfer Distance STD: ", np.around(std_cscore, decimals=3))
    print("Forward Chamfer Distance Mean: ", np.around(avg_f_cscore, decimals=3), "STD: ", np.around(std_f_cscore, decimals=3))
    print("Backward Chamfer Distance Mean: ", np.around(avg_b_cscore, decimals=3), "STD: ", np.around(std_b_cscore, decimals=3))
    print("IoU Mean: ", np.around(avg_iscore, decimals=3), "IoU STD: ", np.around(std_iscore, decimals=3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument("--results_dir", type=str, default="results/", help="Directory containing results to evaluate")
    parser.add_argument("--gt_dir", type=str, default="gt_pcds/", help="Directory for gt ptclouds")

    # Parsing arguments
    args = parser.parse_args()

    evaluate_results(args.results_dir, args.gt_dir)