import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import numpy as np
import transforms3d
import cv2
def campose_to_matrix(rotation):
    pitch, yaw, roll = rotation
    yaw = np.radians(yaw)     
    pitch = np.radians(pitch) 
    roll = np.radians(roll)   
    R_cw = transforms3d.euler.euler2mat(-roll, -pitch, yaw, 'sxyz')
    #R_cw = transforms3d.euler.euler2mat(roll, yaw, pitch, axes='szyx')

    return R_cw
transfer_matrix = np.array(
    [[0,0,1,0],
     [1,0,0,0],
     [0,-1,0,0],
     [0,0,0,1]]
)



campose0= [[-9.99999994e-01, -1.07148030e-04,  3.48220212e-05, -5.32184000e+00],
 [ 4.21914098e-05, -6.95649498e-02,  9.97577424e-01, -7.02800000e-01],
 [ 1.04466063e-04, -9.97577419e-01, -6.95649539e-02,  2.80706000e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

campose1= [[  -0.8660254,    0. ,         -0.5   ,      47.71988  ],
 [  -0.5    ,      0.  ,         0.8660254 , -53.49236  ],
 [   0.    ,      -1.   ,        0.   ,     -126.44252  ],
 [   0.    ,       0.     ,      0.  ,         1.       ]]

campose2= [[-1.0000000e+00 , 0.0000000e+00,  6.1232340e-17,  4.7845390e+01],
 [ 6.1232340e-17,  0.0000000e+00,  1.0000000e+00, -5.3468890e+01],
 [ 0.0000000e+00, -1.0000000e+00,  0.0000000e+00, -1.2643531e+02],
 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]

campose3= [[  -0.8660254 ,   0. ,          0.5,         47.95451  ],
 [   0.5 ,         0.  ,         0.8660254 , -53.53101  ],
 [   0.   ,       -1.   ,        0. ,       -126.43227  ],
 [   0.   ,        0.   ,        0.     ,      1.       ]]

campose4= [[  -0.5   ,       0.  ,         0.8660254 ,  48.00316  ],
 [   0.8660254,    0.  ,         0.5 ,       -53.61789  ],
 [   0.     ,     -1.,           0.  ,      -126.43686  ],
 [   0.      ,     0.     ,      0.   ,        1.       ]]

depth_image = np.load("./figs/obs_depth_0.npy")
# print(depth_image.shape)
# #cv2.imshow(depth_image)
# plt.imshow(depth_image)
# plt.show()

focal_length = 540 
cx = 540
cy = 540  
fx = fy = 540

height, width = depth_image.shape

# points = []

# for i in range(height):
#     for j  in range(width):
#         Z = depth_image[i,j]
#         X = (j - cx) * Z / focal_length
#         Y = (i - cy) * Z / focal_length
#         points.append([X, Y, Z])


# points = np.array(points)  


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# ax.scatter(points[:, 0], points[:, 1], points[:, 2])


# ax.set_xlabel('X right')
# ax.set_ylabel('Y down')
# ax.set_zlabel('Z forward')


# ax.view_init(elev=90, azim=90)  

# plt.show()


x, y = np.meshgrid(np.arange(width), np.arange(height))


Z = depth_image  
X = (x - cx) * Z / focal_length
Y = (y - cy) * Z / focal_length


points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

valid_points = points_3d[Z.flatten() > 0]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)


o3d.visualization.draw_geometries([pcd])






def plot_coordinate_system(ax, rotation_matrix=None, origin=np.array([0, 0, 0]), scale=1.0, 
                          label_prefix='', alpha=1.0, linestyle='-'):
    """
    Plot a coordinate system
    """
    # Unit vectors for the coordinate axes
    # Modified Y-axis direction to point right instead of left
    axes = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * scale
    #axes = np.array([[-1/2, np.sqrt(3)/2, 0], [0, 0, -1], [-np.sqrt(3)/2, -1/2, 0]])
    # Apply rotation if a rotation matrix is provided
    if rotation_matrix is not None:
        axes = np.dot(np.linalg.inv(rotation_matrix),axes)
    
    # Colors for axes
    colors = ['r', 'g', 'b']  # x, y, z correspond to red, green, blue
    
    # Labels
    labels = [f'{label_prefix}X', f'{label_prefix}Y', f'{label_prefix}Z']
    
    # Plot each axis
    for i in range(3):
        ax.quiver(
            origin[0], origin[1], origin[2],  # Start point
            axes[i, 0], axes[i, 1], axes[i, 2],  # Direction vector
            color=colors[i],
            label=labels[i],
            alpha=alpha,
            linestyle=linestyle
        )

def visualize_rotations_separate(pitch, yaw, roll):
    """
    Visualize the coordinate system before and after rotation with given Euler angles
    using separate subplots for clearer comparison
    """
    # Create rotation matrix
    rotation_matrix = np.dot(campose_to_matrix([pitch, yaw, roll]),transfer_matrix[:3,:3])
    #rotation_matrix = campose_to_matrix([pitch, yaw, roll])
    # Set up figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    
    # First subplot - Original coordinate system
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=30, azim=45)
    limit = 1.5
    ax1.set_xlim([-limit, limit])
    ax1.set_ylim([-limit, limit])
    ax1.set_zlim([-limit, limit])
    
    plot_coordinate_system(ax1, scale=1.0, label_prefix='', alpha=1.0)
    
    ax1.set_title('Original Coordinate System')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.legend()
    
    # Second subplot - Rotated coordinate system
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=30, azim=45)
    ax2.set_xlim([-limit, limit])
    ax2.set_ylim([-limit, limit])
    ax2.set_zlim([-limit, limit])
    
    plot_coordinate_system(ax2, rotation_matrix, scale=1.0, label_prefix='', alpha=1.0)
    
    ax2.set_title(f'Rotated Coordinate System\nRoll={roll}, Pitch={pitch}, Yaw={yaw}')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.legend()
    
    # Add title to the entire figure
    fig.suptitle('Coordinate System Before and After Rotation', fontsize=16)
    
    # Add reference notes
    plt.figtext(0.25, 0.02, "Initial coordinate system: X forward, Y right, Z up", 
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Print the rotation matrix
    print("Rotation matrix:")
    print(rotation_matrix)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()
    
    return rotation_matrix
