from enum import unique
from icp import gen_obj_depth
import sim
from random import seed
import os
import camera
from camera import *
import pybullet as p
import numpy as np
import image
import torch
import train_seg_model
from PIL import Image
import torchvision
import torchvision.transforms as transformsim
import icp
import transforms
from scipy.spatial.transform import Rotation
import random
import main
import cv2

# NOTE: using torch version 1.6.0, torchvision version 0.7.0
# check version with: python -c "import torch; print(torch.__version__)"
# install correct version with pip (eg. sudo pip3 install torch==1.6.0)

if __name__ == "__main__":
    random.seed(1)
    color_palette = train_seg_model.get_tableau_palette()

    # Note: Please don't change the order in object_shapes and object_meshes array.
    #   their order is consistent with the trained segmentation model.
    object_shapes = [
        "assets/objects/cube.urdf",
        "assets/objects/rod.urdf",
        "assets/objects/custom.urdf",
    ]
    object_meshes = [
        "assets/objects/cube.obj",
        "assets/objects/rod.obj",
        "assets/objects/custom.obj",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes)
    env.load_gripper()

    # setup camera (this should be consistent with the camera 
    #   used during training segmentation model)
    my_camera = camera.Camera(
        image_size=(480, 640),
        near=0.01,
        far=10.0,
        fov_w=50
    )
    camera_target_position = (env._workspace1_bounds[:, 0] + env._workspace1_bounds[:, 1]) / 2
    camera_target_position[2] = 0
    camera_distance = np.sqrt(((np.array([0.5, -0.5, 0.8]) - camera_target_position)**2).sum())
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=90,
        pitch=-60,
        roll=0,
        upAxisIndex=2,
    )

    # Prepare model (again, should be consistent with segmentation training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3  # RGB
    n_classes = len(object_shapes) + 1  # number of objects + 1 for background class
    model = train_seg_model.miniUNet(n_channels, n_classes)
    model.to(device)
    model, _, _ = train_seg_model.load_chkpt(model, 'checkpoint.pth.tar', device)
    model.eval()

    obj_ids = env._objects_body_ids  # everything else will be treated as background

    # For printing purposes
    object_list = ["Cube", "Rod", "Alien Toothbrush"]
    num_grasped = 0

    is_grasped = np.zeros(3).astype(np.bool)
    while not np.all(is_grasped):  # Keep repeating until the tote is empty
        # ========================= IMAGE CAPTURE & SEGMENTATION =============================
        # Capture rgb and depth image of the tote.
        print("======================================================")
        rgb_obs, depth_obs, _ = camera.make_obs(my_camera, view_matrix)

        # DONE: now generate the segmentation prediction from the model
        pred = None  # pred should contain the predicted segmentation mask
        
        # Use same normalization as model training
        mean_rgb = [0.485, 0.456, 0.406]
        std_rgb = [0.229, 0.224, 0.225]

        # Convert numpy array (image) to tensor, normalize, and add extra dimension (like creating batch of size 1)
        transform = transformsim.Compose([transformsim.ToTensor(), transformsim.Normalize(mean_rgb, std_rgb)])
        img_tensor = transform(rgb_obs).unsqueeze(0).to(device)

        # Run image through model, get prediction (object id per pixel)
        output = model(img_tensor)
        _, pred = torch.max(output, dim=1)

        # Output which objects are found in the image
        unique_ids = torch.unique(pred)
        obj_ids_in_image = unique_ids.tolist()
        objects_in_image = []
        for id in obj_ids_in_image:
            if id == 0:
                continue
            objects_in_image.append(object_list[id-1])
        print(f"\nFound these objects in image: {objects_in_image}")
        
        # Save predicted mask in a folder
        dump_dir = './clear_bin_seg/'
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        for i in range(pred.shape[0]):
            pred_image = train_seg_model.convert_seg_split_into_color_image(pred[0].cpu().numpy())
            cv2.imwrite(f"{dump_dir}/pred_{num_grasped}.png", pred_image.astype(np.uint8))
        
        # =========================== MASK TO WORLD POINT CLOUD ===========================
        markers = []
        # Points in each point cloud to use for ICP.
        #   Adjust this as per your machine performance.
        num_sample_pts = 100

        # Randomly choose an object index to grasp which is not grasped yet.
        # [optional] You can also try out some heuristics to choose which object to grasp.
        #    For example: grasp object which is most isolated to avoid collision with other objects
        obj_index = np.random.choice(np.where(~is_grasped)[0], 1)[0]
        print(f"\nPicking up [{object_list[obj_index]}], Index: {str(obj_index)}")

        # DONE: Mask out the depth based on predicted segmentation mask of object.
        obj_depth = np.zeros_like(depth_obs)
        obj_mask = np.where(pred == obj_index+1, 1, 0)
        obj_depth = depth_obs * obj_mask # Masks depth image based on obj_mask
        # DONE: transform depth to 3d points in camera frame. We will refer to these points as
        #   segmented point cloud or seg_pt_cloud.
        cam_pts = np.zeros((0,3))
        cam_pts = np.asarray(transforms.depth_to_point_cloud(my_camera.intrinsic_matrix, obj_depth[0]))
        if cam_pts.shape == (0,):
            print("No points are present in segmented point cloud. Please check your code. Continuing ...")
            continue

        # DONE: transform 3d points (seg_pt_cloud) in camera frame to the world frame
        world_pts = np.zeros((0,3))
        world_pts = transforms.transform_point3s(cam_view2pose(view_matrix), cam_pts)
        
        # =========================== POINT CLOUD ALIGNMENT (ICP) ===========================
        seg_pt_cloud = world_pts[np.random.choice(range(world_pts.shape[0]), num_sample_pts), :]
        # (optional) Uncomment following to visualize points as small red spheres.
        #   These should approximately lie on chosen object index
        for position in seg_pt_cloud:
             markers.append(sim.SphereMarker(position=position, radius=0.001, rgba_color=[1, 0, 0, 0.8]))

        # Sample points from ground truth mesh. 
        # DONE: sample pts from known object mesh. Use object_shapes[obj_index] - use object_meshes[obj_index]
        #   to locate path of the mesh.
        # - We will call these points ground truth point cloud or gt_pt_cloud.
        # - Hint: use icp.mesh2pts function from hw2
        gt_pt_cloud = icp.mesh2pts(object_meshes[obj_index], len(seg_pt_cloud))
        
        # DONE: Align ground truth point cloud (gt_pt_cloud) to segmented 
        #   point cloud (seg_pt_cloud) using ICP.
        # - Hint: use icp.align_pts function from hw2
        transform = None  # should contain the transformation matrix for transforming ground truth object point cloud to the segmented object point cloud.
        transformed = None # should contain transformed ground truth point cloud
        transform, transformed = icp.align_pts(gt_pt_cloud, seg_pt_cloud, max_iterations=20, threshold=1e-05)

        # (optional) Uncomment following to visualize transformed points as small black spheres.
        #   These should approximately lie on chosen object index
        for position in transformed:
             markers.append(sim.SphereMarker(position=position, radius=0.001, rgba_color=[0, 0, 0, 0.8]))

        # ====================== GRASP POSITION AND ANGLE EXTRACTION ======================
        # DONE: extract grasp position and angle
        position = None  # This should contain the grasp position
        grasp_angle = None  # This should contain the grasp angle
        
        print("\nObject Transform:")
        print(transform)
        rot_matrix = transform[:3,:3]
        translation_matrix = transform[:,3][:3]

        # Get yaw (z rotation) from rotation matrix
        r = Rotation.from_matrix(rot_matrix)
        grasp_angle = r.as_euler('zyx', degrees=False)[0]

        position = translation_matrix
       
        # visualize grasp position using a big red sphere
        markers.append(sim.SphereMarker(position, radius = 0.02))

        # ====================== GRASPING, PATH PLANNING (RRT) & EXECUTION ========================
        # attempt grasping
        grasp_success = env.execute_grasp(position, grasp_angle)
        print(f"\nGrasp success: {grasp_success}")

        if grasp_success:  # Move the object to another tote
            is_grasped[obj_index] = True
            num_grasped+=1

            # Get a list of robot configurations in small step sizes
            path_conf = main.rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, main.MAX_ITERS, main.delta_q, 0.5, env)
            if path_conf is None:
                print("no collision-free path is found within the time budget. continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                main.execute_path(path_conf, env)
        del markers
        p.removeAllUserDebugItems()
        env.robot_go_home()
