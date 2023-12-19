

import argparse
import csv
import glob
import os
import shutil
import sys
import itertools

# Importing Pytorch before Open3D can cause unknown "invalid pointer" error
import open3d as o3d
sys.path.append(os.path.join(os.path.dirname(__file__), '../../cleargrasp/'))
from api import depth_completion_api
from api import utils as api_utils

import attrdict
import imageio
import termcolor
import yaml
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt


def make_depth_model():
    # Load Config File
    CONFIG_FILE_PATH = "depth_completion_files/config_hope.yaml"
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = attrdict.AttrDict(config_yaml)

    results_dir = "results/"

    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)

    depthcomplete = depth_completion_api.DepthToDepthCompletion(normalsWeightsFile=config.normals.pathWeightsFile,
                                                                outlinesWeightsFile=config.outlines.pathWeightsFile,
                                                                masksWeightsFile=config.masks.pathWeightsFile,
                                                                normalsModel=config.normals.model,
                                                                outlinesModel=config.outlines.model,
                                                                masksModel=config.masks.model,
                                                                depth2depthExecutable=config.depth2depth.pathExecutable,
                                                                outputImgHeight=outputImgHeight,
                                                                outputImgWidth=outputImgWidth,
                                                                fx=int(config.depth2depth.fx),
                                                                fy=int(config.depth2depth.fy),
                                                                cx=int(config.depth2depth.cx),
                                                                cy=int(config.depth2depth.cy),
                                                                filter_d=config.outputDepthFilter.d,
                                                                filter_sigmaColor=config.outputDepthFilter.sigmaColor,
                                                                filter_sigmaSpace=config.outputDepthFilter.sigmaSpace,
                                                                maskinferenceHeight=config.masks.inferenceHeight,
                                                                maskinferenceWidth=config.masks.inferenceWidth,
                                                                normalsInferenceHeight=config.normals.inferenceHeight,
                                                                normalsInferenceWidth=config.normals.inferenceWidth,
                                                                outlinesInferenceHeight=config.normals.inferenceHeight,
                                                                outlinesInferenceWidth=config.normals.inferenceWidth,
                                                                min_depth=config.depthVisualization.minDepth,
                                                                max_depth=config.depthVisualization.maxDepth,
                                                                tmp_dir=results_dir)
    return depthcomplete, config



def complete_depth(image, depth, intrinsics, depthcomplete, config):

    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)

    depth = depth.astype(np.float32)
    output_depth, filtered_output_depth = depthcomplete.depth_completion(
                image,
                depth,
                inertia_weight=float(config.depth2depth.inertia_weight),
                smoothness_weight=float(config.depth2depth.smoothness_weight),
                tangent_weight=float(config.depth2depth.tangent_weight),
                mode_modify_input_depth=config.modifyInputDepth.mode,
                dilate_mask=True)

    occ, normal = depthcomplete._return_occ_normals()

    new_intrinsics = np.eye(3)
    new_intrinsics[0,0] = int(config.depth2depth.fx)
    new_intrinsics[1,1] = int(config.depth2depth.fy)
    new_intrinsics[0,2] = int(config.depth2depth.cx)
    new_intrinsics[1,2] = int(config.depth2depth.cy)

    return output_depth, new_intrinsics, outputImgHeight, outputImgWidth, occ, normal

    