
import open3d as o3d
import numpy as np
import math

WIDTH = 2*640
HEIGHT = 2*480

def save_view_point(pcd_ls, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT)
    for pcd in pcd_ls:
        vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd_ls, json_fname, out_fname='renders/test.png'):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(json_fname)
    for pcd in pcd_ls:
        vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.capture_screen_image(out_fname)

    vis.destroy_window()



def convert_realsense_rgb_depth_to_o3d_pcl(
    rgb: np.ndarray, depth: np.ndarray, color_camera_k: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    # NOTE: Assumes that the realsense values (depth, camera_k) are in mm.
    @param rgb (H, W, 3): RGB image from realsense
    @param depth (H, W, 1): Depth image from realsense (in mm, i.e. unchanged)
    @param color_camera_k (3, 3): Color Camera intrinsics from realsense (in mm, i.e. unchanged)
    @return: Open3D point cloud
    """
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1000, convert_rgb_to_intensity=False
    )
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d_rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=float(color_camera_k[0, 0]),
            fy=float(color_camera_k[1, 1]),
            cx=float(color_camera_k[0, 2]),
            cy=float(color_camera_k[1, 2]),
        ),
    )
    return o3d_pcd

def point_cloud_to_depth(
    intrinsics: np.ndarray, camera_points: np.ndarray, color_list: np.ndarray, height: int, width: int
) -> np.ndarray:
    if intrinsics.shape != (3, 3):
        raise ValueError("Invalid input intrinsics")
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError("Invalid camera point")

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    # Project all points at once
    projected_x = np.round((camera_points[:, 0] * fu / camera_points[:, 2]) + u0).astype(int)
    projected_y = np.round((camera_points[:, 1] * fv / camera_points[:, 2]) + v0).astype(int)

    valid_points_mask = (projected_x >= 0) & (projected_x < width) & (projected_y >= 0) & (projected_y < height)

    depth_image = np.zeros((height, width))
    color_image = np.zeros((height, width, 3))

    for i in range(camera_points.shape[0]):
        x = projected_x[i]
        y = projected_y[i]
        if valid_points_mask[i]:
            if depth_image[y, x] == 0 or camera_points[i, 2] < depth_image[y, x]:
                depth_image[y, x] = camera_points[i, 2]
                color_image[y, x, :] = color_list[i]

    return depth_image, color_image

def generate_background_mesh(color_transformed_removed, depth_transformed, PPX, PPY, Fx, Fy):

    color_k = np.eye(3)
    color_k[0,0] = Fx
    color_k[1,1] = Fy
    color_k[0,2] = PPX
    color_k[1,2] = PPY
    print(color_k)
    print(color_transformed_removed)
    print(depth_transformed)
    print(color_k.dtype)
    print(color_transformed_removed.dtype)
    print(depth_transformed.dtype)
    print(np.shape(color_k))
    print(np.shape(color_transformed_removed))
    print(np.shape(depth_transformed))
    pcd_color = convert_realsense_rgb_depth_to_o3d_pcl((color_transformed_removed * 255).astype(np.uint8), depth_transformed * 1000, color_k)

    colors = np.asarray(pcd_color.colors)
    colors[:, [2, 0]] = colors[:, [0, 2]]
    pcd_color.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd_color])

    mask = np.any(color_transformed_removed != 0, axis = -1)

    empty = np.ones((np.shape(mask)[0], np.shape(mask)[1], 1))

    idx = np.argwhere(empty)
    points_3D = []
    points_3D_further = []
    for index in idx:
        if index[0] % 10 == 0 and index[1] % 10 == 0:
            if depth_transformed[index[0], index[1]] != 0:
                point_3D = np.array(
                    [
                        (index[1] - PPX) * depth_transformed[index[0], index[1]] / Fx,
                        (index[0] - PPY) * depth_transformed[index[0], index[1]] / Fy,
                        depth_transformed[index[0], index[1]],
                    ]
                )
                points_3D.append(point_3D)

                for x in range(2,100):
                    point_3D_further = np.array(
                        [
                            (index[1] - PPX) * (depth_transformed[index[0], index[1]] + (x * 0.01)) / Fx,
                            (index[0] - PPY) * (depth_transformed[index[0], index[1]] + (x * 0.01)) / Fy,
                            depth_transformed[index[0], index[1]] + (x * 0.01),
                        ]
                    )
                    points_3D_further.append(point_3D_further)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)

    pcd_further = o3d.geometry.PointCloud()
    pcd_further.points = o3d.utility.Vector3dVector(points_3D_further)

    pcd_further.estimate_normals()

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_further, depth=9)

    vertices_to_remove = densities < np.quantile(densities, 0.15)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.paint_uniform_color([0.5, 0.5, 0.5])

    return mesh


def depth_to_point_cloud(intrinsics: np.ndarray, depth_image: np.ndarray, color_array: np.ndarray) -> np.ndarray:
    """
    Back project a depth image to a point cloud.
    Note: Only output those points whose depth > 0.
    FIXME: This can probably be sped up by a lot using vectorization.
    @param intrinsics ([3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
    @param depth_image ([H, W]): depth image
    @return: pcl ([N, 3])
    """
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    point_cloud = []
    color_list = []
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] <= 0:
                continue
            point_cloud.append(
                np.array(
                    [
                        (u - u0) * depth_image[v, u] / fu,
                        (v - v0) * depth_image[v, u] / fv,
                        depth_image[v, u],
                    ]
                )
            )
            color_list.append(color_array[v, u])
    point_cloud = np.array(point_cloud)
    color_list = np.array(color_list)
    return point_cloud, color_list

def point_cloud_to_depth(
    intrinsics: np.ndarray, camera_points: np.ndarray, color_list: np.ndarray, height: int, width: int
) -> np.ndarray:
    """
    Project points in camera space to the image plane.
    FIXME: This can probably be sped up by a lot using vectorization.
    @param intrinsics:  ([3, 3]): Pinhole intrinsics.
    @param camera_points: ([N, 3]): N 3D points (x, y, z) in camera coordinates.
    @param height: (int): Height of the image.
    @param width: (int): Width of the image.
    @return: depth_image [height, width]
    @raises: ValueError: If intrinsics are not the correct shape.
    @raises: ValueError: If camera points are not the correct shape.
    """
    if intrinsics.shape != (3, 3):
        raise ValueError("Invalid input intrinsics")
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError("Invalid camera point")

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    image = np.zeros((height, width))

    color_image = np.zeros((height, width, 3))

    color_image = np.empty((height, width, 3))
    color_image[:] = np.nan

    last_z = np.zeros((height, width))

    for i in range(camera_points.shape[0]):
        w = int(np.round((camera_points[i, 0] * fu / camera_points[i, 2]) + u0))
        h = int(np.round((camera_points[i, 1] * fv / camera_points[i, 2]) + v0))
        if h > 0 and h < height and w > 0 and w < width:
            if np.isnan(color_image[h, w, 0]):
                image[h, w] = camera_points[i, 2]
                color_image[h, w, :] = color_list[i,:]
                last_z[h, w] = camera_points[i, 2]
            else:
                if camera_points[i, 2] < last_z[h, w]:
                    image[h, w] = camera_points[i, 2]
                    color_image[h, w, :] = color_list[i,:]
    return image, color_image


def transform_points_helper(points, homogeneous_trans_mat):  # points(n, 3)
    points_homogeneous = np.concatenate(
        [points.T, np.ones((1, points.shape[0]))], axis=0
    )
    points_homogeneous = homogeneous_trans_mat @ points_homogeneous
    points = points_homogeneous[:3, :] / points_homogeneous[3, :]
    points = points.T
    return points

def transform_depth(depth, color, depth_camera_k, color_camera_k, ir1_to_color, new_w=None, new_h=None):
    # Alright, the logic is honestly straightforward.
    # - Basically, convert depth to point-cloud using depth_camera_k
    pcl, color_list = depth_to_point_cloud(depth_camera_k, depth, color)
    # Then transform the pcl using ir1_to_color
    pcl = transform_points_helper(pcl, ir1_to_color)
    # Then convert that pcl to depth using color_camera_k
    if new_w == None:
        depth_transformed, color_transformed = point_cloud_to_depth(
            color_camera_k, pcl, color_list, depth.shape[0], depth.shape[1]
        )
    else:
        depth_transformed, color_transformed = point_cloud_to_depth(
            color_camera_k, pcl, color_list, new_w, new_h
        )
    return depth_transformed, color_transformed


def sample_hemisphere_point_uniform(radius=1.0, phi=0, theta=0):

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = -radius * math.cos(theta)
    
    return (x, y, z)


def look_at(camera_position, target_position, up_vector=np.array([0, 1, 0])):
    # Calculate the forward vector (the direction from the camera to the target)
    forward = target_position - camera_position
    forward /= np.linalg.norm(forward)
    # forward = -forward

    # Calculate the right vector (perpendicular to the forward and up vectors)
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)

    # Calculate the up vector (perpendicular to the forward and right vectors)
    up = np.cross(forward, right)

    # Create the view matrix
    view_matrix = np.identity(4)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = up
    view_matrix[:3, 2] = forward
    view_matrix[:3, 3] = camera_position

    # Return the view matrix
    return view_matrix