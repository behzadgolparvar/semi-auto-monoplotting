import numpy as np
from numpy import pi
import cv2
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
import os
import torch
from osgeo import gdal



def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
  dim = None
  (h, w) = image.shape[:2]

  if width is None and height is None:
      return image
  if width is None:
      r = height / float(h)
      dim = (int(w * r), height)
  else:
      r = width / float(w)
      dim = (width, int(h * r))

  return cv2.resize(image, dim, interpolation=inter)


def readImage_withScale(Path='.', scalefactor=1):
  img = cv2.imread(Path)
  img2=ResizeWithAspectRatio(img, width=int(img.shape[1]*scalefactor))
  return img2


def showImg_and_KeyPoints(
        img, img_keypoints, pointfill=True, fillcolor='red',
        showGrid=True, showLegend=False, fontsize=14,
        labelString='key points', showPointNumber=True,
        pointsize=50, figuresize=(12, 15), showTicks=True, colormap='viridis',
        saveTofile=False, filePath='./output.png', dpi=300,
        ax=None):
  """
  Display an image with overlaid keypoints.

  Parameters:
      img (array): The image to display.
      img_keypoints (array): Array of keypoints (Nx2) to overlay on the image.
      pointfill (bool): Whether to fill the points.
      fillcolor (str): Color of the points.
      showGrid (bool): Whether to show grid lines.
      showLegend (bool): Whether to display a legend.
      fontsize (int): Font size for the legend and annotations.
      labelString (str): Label for the keypoints in the legend.
      showPointNumber (bool): Whether to display point numbers.
      pointsize (int): Size of the points.
      figuresize (tuple): Size of the figure (if `ax` is not provided).
      colormap (str): Colormap for grayscale images.
      saveTofile (bool): Whether to save the output to a file.
      filePath (str): Path to save the file.
      dpi (int): DPI for saving the file.
      ax (matplotlib.axes.Axes): Axes object for plotting (optional).
  """
  # If no axis is provided, create a new figure and axiss
  if ax is None:
      fig, ax = plt.subplots(figsize=figuresize)

  # Display the image
  if len(img.shape) == 3:  # RGB image
      ax.imshow(img[:, :, ::-1])
  elif len(img.shape) == 2:  # Grayscale image
      ax.imshow(img, cmap=colormap)

  # Plot keypoints
  if pointfill:
      ax.scatter(img_keypoints[:, 0], img_keypoints[:, 1],
                  s=pointsize, c=fillcolor, label=labelString)
  else:
      ax.scatter(img_keypoints[:, 0], img_keypoints[:, 1],
                  s=pointsize, facecolors='none', edgecolors='b',
                  alpha=1, label=labelString)

  width_img = img.shape[1]
  height_img = img.shape[0]

  ax.axis('scaled')
  ax.set_xlim(0, width_img)
  # ax.set_ylim(0, height_img)
  # ax.invert_yaxis()

  if not showTicks:
    ax.set_xticks([])
    ax.set_yticks([])

  # Show grid if requested
  if showGrid:
      ax.grid(which='major')
      ax.minorticks_on()
      ax.grid(which='minor', linestyle=':')

  # Show legend if requested
  if showLegend:
      ax.legend(fontsize=fontsize)

  # Annotate points with numbers if requested
  if showPointNumber:
      for i, (x, y) in enumerate(img_keypoints):
          margin = -12  # Adjust as needed
          ax.text(x, y + margin, str(i), horizontalalignment='center',
                  size='small', color='blue', weight='semibold')

  # Save to file if requested
  if saveTofile:
      plt.savefig(filePath, bbox_inches='tight', dpi=dpi)

  # Show the figure if no axis was provided
  if ax is None:
    plt.show()


def showImg_and_KeyPoints_and_projectedPoints(img, img_keypoints, projectedPoints, showGrid=True, showLegend=False,
                                              fontsize=14, showPointNumber=True, pointsize=50, showTicks=True, figuresize=(12,15),
                                              colormap='viridis', saveTofile=False, filePath='./output.png', dpi=300,
                                              ax=None):
  if ax is None:
    fig, ax = plt.subplots(figsize=figuresize)

  
  if len(img.shape) == 3:
    ax.imshow(img[:,:,::-1])
  elif len(img.shape) == 2:
     ax.imshow(img, cmap=colormap)

  
  ax.scatter(img_keypoints[:,0], img_keypoints[:,1], s=30, c='r', label='Image Key Points')
  ax.scatter(projectedPoints[:,0], projectedPoints[:,1], s=pointsize,  facecolors='none', edgecolors='b',
              alpha=1, label='Projected DEM Key Points')
  
  width_img = img.shape[1]
  height_img = img.shape[0]

  ax.axis('scaled')
  ax.set_xlim(0, width_img)
  # ax.set_ylim(0, height_img)
  # ax.invert_yaxis()
  
  if showGrid:
    ax.grid(which='major')
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':')
  if showLegend:
    ax.legend(fontsize=fontsize)

  if not showTicks:
    ax.set_xticks([])
    ax.set_yticks([])
     

  if showPointNumber:
    for i in range(img_keypoints.shape[0]):
            margin = -12 ## 15*((-1)**i) #Selected_Image_keypoints[i,0]*((-1)**i)*0.005
            ax.text(img_keypoints[i,0], img_keypoints[i,1]+margin, str(i),
                    horizontalalignment='center', size='small', color='red', weight='semibold')
            
    for i in range(projectedPoints.shape[0]):
        margin = 20 ## 15*((-1)**i) #Selected_Image_keypoints[i,0]*((-1)**i)*0.005
        ax.text(projectedPoints[i,0], projectedPoints[i,1]+margin, str(i),
                horizontalalignment='center', size='small', color='b', weight='semibold')
  
  if saveTofile == True:
    plt.savefig(filePath, bbox_inches='tight', dpi=dpi)

  if ax is None:
    plt.show()
  return


def TorchRotationMatrix(axis: str, angle):
  cos = torch.cos(angle)
  sin = torch.sin(angle)
  one = torch.ones_like(angle)
  zero = torch.zeros_like(angle)

  if axis == "X":
    R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
  if axis == "Y":
    R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
  if axis == "Z":
    R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
  return torch.stack(R_flat, -1).reshape((3, 3))


def TorchEulerAnglesToRotationMatrixDegree(THETA_rad):
  R_x = TorchRotationMatrix("X", THETA_rad[0])
  R_y = TorchRotationMatrix("Y", THETA_rad[1])
  R_z = TorchRotationMatrix("Z", THETA_rad[2])
  return torch.matmul(R_z, torch.matmul(R_y,R_x))


def processDEMraster(filePath):
  gdal.UseExceptions()
  gdata = gdal.Open(filePath)
  gdal_band = gdata.GetRasterBand(1)
  nodataval = gdal_band.GetNoDataValue()
  DEM_DataArray = gdata.ReadAsArray().astype('float64')
  # replace missing values if necessary
  if np.any(DEM_DataArray == nodataval):
    DEM_DataArray[DEM_DataArray == nodataval] = np.nan
  
  GDAL_GeoTransform = gdata.GetGeoTransform()
  return DEM_DataArray, GDAL_GeoTransform
   

def DEMrasterKeyPointsToGeoKeyPoints(RasterKeyPoints, DEM_DataArray, GDAL_GeoTransform):
  DEMRasterKeyPoints_indices = RasterKeyPoints.round(0).astype('int64')
  GeoKeyPoints = []
  for i_horizental,j_vertical in DEMRasterKeyPoints_indices:
    x= (i_horizental)*GDAL_GeoTransform[1] + (j_vertical)*GDAL_GeoTransform[2] + GDAL_GeoTransform[0]
    y= (i_horizental)*GDAL_GeoTransform[4] + (j_vertical)*GDAL_GeoTransform[5] + GDAL_GeoTransform[3]
    z= DEM_DataArray[j_vertical,i_horizental]
    GeoKeyPoints.append([x,y,z])
    
  GeoKeyPoints = np.array(GeoKeyPoints)
  return GeoKeyPoints


def createGeoPointCloudfromDEM(filePath='./input', outputPath='./output', saveoutput=True):
  DEM_DataArray, GDAL_GeoTransform = processDEMraster(filePath)
   
  GeoPoints3D = []
  for j_vertical in range(DEM_DataArray.shape[0]):
      for i_horizental in range(DEM_DataArray.shape[1]):
          
          x= (i_horizental)*GDAL_GeoTransform[1] + (j_vertical)*GDAL_GeoTransform[2] + GDAL_GeoTransform[0]
          y= (i_horizental)*GDAL_GeoTransform[4] + (j_vertical)*GDAL_GeoTransform[5] + GDAL_GeoTransform[3]
          z= DEM_DataArray[j_vertical,i_horizental]
          GeoPoints3D.append([x,y,z])
  GeoPoints3D = np.array(GeoPoints3D)

  if saveoutput:
    np.savetxt(outputPath, GeoPoints3D, delimiter=',')

  return GeoPoints3D


def PointCloudVisualizer(PointCloudFilePath, width=1500, height=950):

  points = np.loadtxt(PointCloudFilePath, delimiter=',')
  pointcloud = o3d.geometry.PointCloud()
  pointcloud.points = o3d.utility.Vector3dVector(points)

  vis = o3d.visualization.Visualizer()
  left = 0 #int(0.5 * (1512 - width))
  vis.create_window('Open3D', width, height,
                    left, top=35, visible=True)
  vis.add_geometry(pointcloud)

  RenderOpt = vis.get_render_option()
  RenderOpt.show_coordinate_frame= True
  RenderOpt.mesh_show_wireframe = True
  vis.run()
  vis.destroy_window()


def createDEMimage(inputDEMPath, outputImagePath):
  DEM_DataArray,_ = processDEMraster(inputDEMPath)
  DEM_Image = ((DEM_DataArray/DEM_DataArray.max())*255).astype('uint8')
  cv2.imwrite(outputImagePath, DEM_Image)
  return


def showDEM3D_and_KeyPoints(DEM3D, img_keypoints, pointfill=True, showGrid=True, showPointNumber=True, pointsize=50, figuresize=(12,15), colormap='viridis'):
  
  fig = plt.figure(figsize=figuresize)
  
  plt.scatter(DEM3D[:,0],
              DEM3D[:,1],
              c=DEM3D[:,2])

  if pointfill:
    plt.scatter(img_keypoints[:,0], img_keypoints[:,1], s=pointsize, c='r', label='Key Points')
  else:
    plt.scatter(img_keypoints[:,0], img_keypoints[:,1], s=pointsize, facecolors='none', edgecolors='b', alpha=1, label='Key Points')

  plt.axis('scaled')

  if showGrid:
    plt.grid(which='major')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.legend(fontsize=14)
  if showPointNumber:
    for i in range(img_keypoints.shape[0]):
            margin = +250 ## 15*((-1)**i) #Selected_Image_keypoints[i,0]*((-1)**i)*0.005
            plt.text(img_keypoints[i,0], img_keypoints[i,1]+margin, str(i),
                    horizontalalignment='center', size='small', color='blue', weight='semibold')
  plt.show()
  return


def forwardMappingImageDistortion(img, distortionStrength):
    H, W,_ = img.shape
    center_y, center_x,  = (H - 1) / 2.0, (W - 1) / 2.0
    distorted_img = np.full((H, W, 3), np.nan, dtype='float64')

    
    for x_i in range(W):
        for y_j in range(H):

            dy = y_j - center_y
            dx = x_i - center_x

            distance = dx**2 + dy**2
            r = 1 + (distance * distortionStrength)

            dy_distorted = dy * r
            dx_distorted = dx * r
            
            y_j_distorted = np.round(dy_distorted + center_y).astype('int64')
            x_i_distorted = np.round(dx_distorted + center_x).astype('int64')


            if (0 <= x_i_distorted < W) & (0 <= y_j_distorted < H):
                distorted_img[y_j_distorted, x_i_distorted, :] = img[y_j, x_i, :]
            else:
                pass

    mask = np.isnan(distorted_img[:,:,0]).astype(np.uint8)
    distorted_img_inpaint = distorted_img.copy()
    distorted_img_inpaint[np.isnan(distorted_img_inpaint)] = 0
    distorted_img_inpaint = distorted_img_inpaint.astype(np.uint8)
    distorted_img_inpaint = cv2.inpaint(distorted_img_inpaint, mask, 3, cv2.INPAINT_TELEA)

    return distorted_img_inpaint, distorted_img


def forwardMappingPointsDistortion(img, imgPoints, distortionStrength):
  H, W,_ = img.shape
  center_y, center_x,  = (H - 1) / 2.0, (W - 1) / 2.0
  
  dxdy_points = imgPoints - np.array([center_x, center_y])

  distance = np.sum(dxdy_points**2, axis=1, keepdims=True)
  r = 1 + (distance * distortionStrength)
  dxdy_points_distorted = np.multiply(dxdy_points, r)
  distorted_points = dxdy_points_distorted + np.array([center_x, center_y])
  
  in_img_filter = ((distorted_points >= np.array([0, 0])) & (distorted_points < np.array([W, H]))).all(axis=1)
  distorted_points_in_image = distorted_points[in_img_filter]


  return distorted_points_in_image


def forwardMappingPointsDistortionTorch(img, img_points, distortion_strength):
    """
    Apply forward mapping with distortion to points using PyTorch.

    Args:
        img (torch.Tensor): The input image tensor of shape (H, W, C).
        img_points (torch.Tensor): Points to be distorted of shape (N, 2).
        distortion_strength (float): The strength of distortion applied.

    Returns:
        torch.Tensor: Distorted points that remain within the image.
    """
    H, W, _ = img.shape
    center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0

    # Shift points relative to the center
    dxdy_points = img_points - torch.tensor([center_x, center_y], device=img_points.device)

    # Compute squared distance and distortion factor
    distance = torch.sum(dxdy_points**2, dim=1, keepdim=True)
    r = 1 + (distance * distortion_strength)

    # Apply distortion
    dxdy_points_distorted = dxdy_points * r
    distorted_points = dxdy_points_distorted + torch.tensor([center_x, center_y], device=img_points.device)

    # Filter points within the image bounds
    in_img_filter = ((distorted_points >= torch.tensor([0, 0], device=img_points.device)) & 
                     (distorted_points < torch.tensor([W, H], device=img_points.device))).all(dim=1)
    distorted_points_in_image = distorted_points[in_img_filter]

    return distorted_points


def inverseMappingImageDistortion(img, distortionStrength):
    H, W,_ = img.shape
    center_y, center_x,  = (H - 1) / 2.0, (W - 1) / 2.0
    distorted_img = np.full((H, W, 3), np.nan, dtype='float64')

    
    for x_i_distorted in range(W):
        for y_j_distorted in range(H):

            dy_distorted = y_j_distorted - center_y
            dx_distorted = x_i_distorted - center_x

            distance = dx_distorted**2 + dy_distorted**2
            r = 1 + (distance * distortionStrength)

            dx = dx_distorted/r
            dy = dy_distorted/r
            
            y_j = np.round(dy + center_y).astype('int64')
            x_i = np.round(dx + center_x).astype('int64')


            if (0 <= x_i < W) & (0 <= y_j < H):
                distorted_img[y_j_distorted, x_i_distorted, :] = img[y_j, x_i, :]
            else:
                pass
            
    return distorted_img


def DEMrasterKeypointFinder(DEMraster, orb_params=None):
    """
    Detect keypoints and compute descriptors for a DEM raster using ORB.

    Args:
        DEMraster: Input DEM raster image.
        orb_params: Dictionary of ORB parameters to override defaults.

    Returns:
        orb_keypoints: List of keypoints detected by ORB.
        orb_des: Descriptors of the keypoints.
    """
    default_orb_params = {
        "nfeatures": 1000,
        "scaleFactor": 1.9,
        "nlevels": 8,
        "edgeThreshold": 31,
        "firstLevel": 0,
        "WTA_K": 4,
        "scoreType": 0,
        "patchSize": 31,
        "fastThreshold": 10
    }

    # Update default parameters with user-provided ones
    if orb_params:
        default_orb_params.update(orb_params)

    orb = cv2.ORB_create(
        nfeatures=default_orb_params["nfeatures"],
        scaleFactor=default_orb_params["scaleFactor"],
        nlevels=default_orb_params["nlevels"],
        edgeThreshold=default_orb_params["edgeThreshold"],
        firstLevel=default_orb_params["firstLevel"],
        WTA_K=default_orb_params["WTA_K"],
        scoreType=default_orb_params["scoreType"],
        patchSize=default_orb_params["patchSize"],
        fastThreshold=default_orb_params["fastThreshold"]
    )

    # Detect and compute keypoints and descriptors
    orb_keypoints, orb_descriptors = orb.detectAndCompute(DEMraster, None)
    
    if len(orb_keypoints) == 0: print("no keypoint founded")
    else:
       keypoints = np.array([i.pt for i in orb_keypoints])
       descriptors = orb_descriptors

    return keypoints, descriptors


class ProjectModel(torch.nn.Module):
  def __init__(self, cx, cy, cz, th1, th2, th3, VA, device='cpu'):

    super().__init__()
    self.device = torch.device(device)

    self.Cam_x = torch.nn.Parameter(torch.tensor(cx, dtype=torch.float64, requires_grad=True))
    self.Cam_y = torch.nn.Parameter(torch.tensor(cy, dtype=torch.float64, requires_grad=True))
    self.Cam_z = torch.nn.Parameter(torch.tensor(cz, dtype=torch.float64, requires_grad=True))
    self.theta1 = torch.nn.Parameter(torch.tensor(th1, dtype=torch.float64, requires_grad=True))
    self.theta2 = torch.nn.Parameter(torch.tensor(th2, dtype=torch.float64, requires_grad=True))
    self.theta3 = torch.nn.Parameter(torch.tensor(th3, dtype=torch.float64, requires_grad=True))
    self.ViewAngle = torch.nn.Parameter(torch.tensor(VA, dtype=torch.float64, requires_grad=True))

###########################################################################################

  def forward(self, img, PntCld, dtcCor):
   
    self.PointCloud = torch.from_numpy(PntCld).to(self.device)
    self.detectedCorners = torch.from_numpy(dtcCor).to(self.device)

    self.N_d = self.detectedCorners.shape[0]
    self.N_p = self.PointCloud.shape[0]

    self.width_img = img.shape[1]
    self.height_img = img.shape[0]


    self.THETA = torch.stack((self.theta1, self.theta2, self.theta3), -1)
    self.THETA_rad = self.THETA*(pi/180)


    self.Cam_pos = torch.stack((self.Cam_x, self.Cam_y, self.Cam_z), -1)
    self.tvec = - self.Cam_pos


    self.RotMatrix = TorchEulerAnglesToRotationMatrixDegree(self.THETA_rad).to(self.device)

    self.focal_length = - (self.height_img)/(2*torch.tan((self.ViewAngle /2)*(pi/180.)))

    self.camera_matrix = torch.stack(
         (self.focal_length, torch.tensor(0), torch.tensor(self.width_img/2)-0.5,
          torch.tensor(0), self.focal_length, torch.tensor(self.height_img/2)-0.5,
          torch.tensor(0), torch.tensor(0), torch.tensor(1)) ).reshape((3,3))



    self.Translated = (self.PointCloud + self.tvec)
    self.xyz = torch.bmm(self.RotMatrix.repeat(self.N_p,1).reshape(self.N_p,3,3), self.Translated.reshape(self.N_p,3,1))
    self.xyz_prime = self.xyz / self.xyz[:,2].unsqueeze(1)
    self.uv = torch.bmm(self.camera_matrix.repeat(self.N_p,1).reshape(self.N_p,3,3), self.xyz_prime).reshape(self.N_p,3)


    self.Y_invert = torch.tensor([1, -1, 1], dtype=torch.float64, device=self.device).repeat(self.N_p,1)
    self.Y_invert_height = torch.tensor([0, self.height_img, 0], device=self.device).repeat(self.N_p,1)


    self.uv_final = ((self.uv * self.Y_invert) + self.Y_invert_height)[:,0:2]

    self.Tdet_corners_expanded = self.detectedCorners.reshape(self.N_d, 1, 2).repeat_interleave(self.N_p,dim=1)
    self.Loss = ((self.Tdet_corners_expanded - self.uv_final)**2).sum(dim=2).min(dim=1)[0].sum()


    return self.Loss


class ProjectModelWithDistortion(torch.nn.Module):
  def __init__(self, cx, cy, cz, th1, th2, th3, VA, k_distortion, device='cpu'):

    super().__init__()
    self.device = torch.device(device)

    self.Cam_x = torch.nn.Parameter(torch.tensor(cx, dtype=torch.float64, requires_grad=True))
    self.Cam_y = torch.nn.Parameter(torch.tensor(cy, dtype=torch.float64, requires_grad=True))
    self.Cam_z = torch.nn.Parameter(torch.tensor(cz, dtype=torch.float64, requires_grad=True))
    self.theta1 = torch.nn.Parameter(torch.tensor(th1, dtype=torch.float64, requires_grad=True))
    self.theta2 = torch.nn.Parameter(torch.tensor(th2, dtype=torch.float64, requires_grad=True))
    self.theta3 = torch.nn.Parameter(torch.tensor(th3, dtype=torch.float64, requires_grad=True))
    self.ViewAngle = torch.nn.Parameter(torch.tensor(VA, dtype=torch.float64, requires_grad=True))
    self.Kdist = torch.nn.Parameter(torch.tensor(k_distortion, dtype=torch.float64, requires_grad=True))
###########################################################################################

  def forward(self, img, PntCld, dtcCor):
   
    self.PointCloud = torch.from_numpy(PntCld).to(self.device)
    self.detectedCorners = torch.from_numpy(dtcCor).to(self.device)

    self.N_d = self.detectedCorners.shape[0]
    self.N_p = self.PointCloud.shape[0]

    self.width_img = img.shape[1]
    self.height_img = img.shape[0]


    self.THETA = torch.stack((self.theta1, self.theta2, self.theta3), -1)
    self.THETA_rad = self.THETA*(pi/180)


    self.Cam_pos = torch.stack((self.Cam_x, self.Cam_y, self.Cam_z), -1)
    self.tvec = - self.Cam_pos


    self.RotMatrix = TorchEulerAnglesToRotationMatrixDegree(self.THETA_rad).to(self.device)

    self.focal_length = - (self.height_img)/(2*torch.tan((self.ViewAngle /2)*(pi/180.)))

    self.camera_matrix = torch.stack(
         (self.focal_length, torch.tensor(0), torch.tensor(self.width_img/2)-0.5,
          torch.tensor(0), self.focal_length, torch.tensor(self.height_img/2)-0.5,
          torch.tensor(0), torch.tensor(0), torch.tensor(1)) ).reshape((3,3))



    self.Translated = (self.PointCloud + self.tvec)
    self.xyz = torch.bmm(self.RotMatrix.repeat(self.N_p,1).reshape(self.N_p,3,3), self.Translated.reshape(self.N_p,3,1))
    self.xyz_prime = self.xyz / self.xyz[:,2].unsqueeze(1)
    self.uv = torch.bmm(self.camera_matrix.repeat(self.N_p,1).reshape(self.N_p,3,3), self.xyz_prime).reshape(self.N_p,3)


    self.Y_invert = torch.tensor([1, -1, 1], dtype=torch.float64, device=self.device).repeat(self.N_p,1)
    self.Y_invert_height = torch.tensor([0, self.height_img, 0], device=self.device).repeat(self.N_p,1)


    self.uv_final = ((self.uv * self.Y_invert) + self.Y_invert_height)[:,0:2]

    self.uv_final_distorted = forwardMappingPointsDistortionTorch(img, self.uv_final, self.Kdist)

    self.Tdet_corners_expanded = self.detectedCorners.reshape(self.N_d, 1, 2).repeat_interleave(self.N_p,dim=1)
    self.Loss = ((self.Tdet_corners_expanded - self.uv_final_distorted)**2).sum(dim=2).min(dim=1)[0].sum()


    return self.Loss


def Rotation_Matrix(pitch_x, heading_y, roll_z):

    from numpy.linalg import multi_dot

    theta = np.array([pitch_x, heading_y, roll_z])*(pi/180)
    R_pitch_x = np.array([[1,         0,                  0                ],
                          [0,         np.cos(theta[0]),  -np.sin(theta[0]) ],
                          [0,         np.sin(theta[0]),   np.cos(theta[0]) ] ])

    R_heading_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                            [0,                     1,      0                   ],
                            [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                            ])

    R_roll_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                        [np.sin(theta[2]),    np.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])

    RotationMatrix = multi_dot([ R_roll_z, R_heading_y,  R_pitch_x])   
    
    return RotationMatrix


def OpenGLandCV_to_Open3D_Angles_deg(OpenGLandCV_angles):
    Coordinate_Rotate = Rotation_Matrix(180,0,0).round(1)
    OP3D_angles = np.matmul(Coordinate_Rotate, OpenGLandCV_angles)-np.array([180,0,0])
    return OP3D_angles


def Open3D_to_OpenGLandCV_Angles_deg(OP3D_angles):
    Coordinate_Rotate = Rotation_Matrix(180,0,0).round(1)
    OpenGLCV_angles = np.matmul(Coordinate_Rotate, OP3D_angles)+np.array([180,0,0])
    return OpenGLCV_angles


def Model_View_Matrix(Cam_x, Cam_y, Cam_z, pitch_x, heading_y, roll_z):

    from numpy.linalg import multi_dot

    tvec = - np.array([Cam_x, Cam_y, Cam_z]) 

    M_trans_H = np.identity(4)
    M_trans_H[0:3,-1] = tvec

    theta = np.array([pitch_x,heading_y,roll_z])*(pi/180)

    R_pitch_x = np.array([[1,         0,                  0                ],
                          [0,         np.cos(theta[0]),  -np.sin(theta[0]) ],
                          [0,         np.sin(theta[0]),   np.cos(theta[0]) ] ])


    R_heading_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                            [0,                     1,      0                   ],
                            [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                            ])

    R_roll_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                        [np.sin(theta[2]),    np.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])

    R_pitch_x_H = np.concatenate( (np.concatenate((R_pitch_x, np.array([[0],[0],[0]])), axis=1),
                     np.array([[0,0,0,1]])), axis=0)
    R_heading_y_H = np.concatenate( (np.concatenate((R_heading_y, np.array([[0],[0],[0]])), axis=1),
                     np.array([[0,0,0,1]])), axis=0)
    R_roll_z_H = np.concatenate( (np.concatenate((R_roll_z, np.array([[0],[0],[0]])), axis=1),
                     np.array([[0,0,0,1]])), axis=0)

    ViewMatrix = multi_dot([ R_roll_z_H, R_heading_y_H,  R_pitch_x_H, M_trans_H])   
    ModelMatrix = np.identity(4)
    Model_View_Matrix = np.matmul(ViewMatrix, ModelMatrix)

    return Model_View_Matrix    


def Custom_Visualizer_CameraPose_FOV_Complete(obj,
                                              Cam_x, Cam_y, Cam_z,
                                              pitch_x, heading_y, roll_z,
                                              fov, Width_img, Height_img):
            
    
    Extrinsic_H = Model_View_Matrix(Cam_x, Cam_y, Cam_z,
                  pitch_x, heading_y, roll_z)
    
    
    vis = o3d.visualization.Visualizer()
    vis.create_window('camera_view', Width_img, Height_img,
                      left=int(0.5*(2496-Width_img)), top=55, visible=True)
    vis.add_geometry(obj)

    view_ctl = vis.get_view_control()  # Everything good
    
    fov_add = (fov-60)/5.
    #print(view_ctl.get_field_of_view())
    view_ctl.change_field_of_view(step=fov_add)
    #print(view_ctl.get_field_of_view())
    
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = Extrinsic_H  
    
    view_ctl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    
    RenderOpt = vis.get_render_option()
    RenderOpt.show_coordinate_frame= True
    RenderOpt.mesh_show_wireframe = True
    RenderOpt.mesh_color_option = o3d.visualization.MeshColorOption.ZCoordinate
    
    
    #vis.update_geometry()
    depth = vis.capture_depth_float_buffer(True)
    depth_image = np.asarray(depth)
    z_image = depth_image #* 1000  # Scale if needed
    # plt.imshow(z_image, cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('./zImage2024.jpeg')
    
    img = vis.capture_screen_float_buffer(True)
    
    vis.run()
    vis.destroy_window()
    

    return view_ctl, img, z_image


def Projection_Matrix(ImageHeight, ImageWidth, FOV):
    L = - ImageWidth//2    # when symmetric frustum: L+R=0
    R = ImageWidth//2

    B = - ImageHeight//2
    T = ImageHeight//2   # when symmetric frustum: T+B=0

    N = T/np.tan((FOV/2)*(pi/180.))
    F = 10*N

    Projection_Matrix = np.array(
    [[2*N/(R-L)  ,  0         ,  (R+L)/(R-L),   0         ],
     [0          ,  2*N/(T-B) ,  (T+B)/(T-B),   0         ],
     [0          ,  0         , -(F+N)/(F-N),  -2*F*N/(F-N)],
     [0          ,  0         , -1          ,   0         ]])

    return Projection_Matrix


def Projecting_World_Points(C_x, C_y, C_z, Pitch_x, Heading_y, Roll_z,
                            ImageHeight, ImageWidth, FOV, XYZ_world):

    XYZ_h = np.concatenate(( XYZ_world, np.full(XYZ_world.shape[0],1).reshape(-1,1) ), axis=1)
    ModelView_Mat = Model_View_Matrix(C_x, C_y, C_z, Pitch_x, Heading_y, Roll_z)

    XYZ_eye_h = np.matmul(ModelView_Mat, XYZ_h[:, :, None]).squeeze(-1)

    ProjMat = Projection_Matrix(ImageHeight, ImageWidth, FOV)
    
    XYZ_clip_h = np.matmul(ProjMat, XYZ_eye_h[:, :, None]).squeeze(-1)

    X_clip_bool_1 = (XYZ_clip_h[:,0] > -XYZ_clip_h[:,3])
    X_clip_bool_2 = (XYZ_clip_h[:,0] <  XYZ_clip_h[:,3])
    Y_clip_bool_1 = (XYZ_clip_h[:,1] > -XYZ_clip_h[:,3])
    Y_clip_bool_2 = (XYZ_clip_h[:,1] <  XYZ_clip_h[:,3])    
    
    ClippedVisIndex = np.where(X_clip_bool_1*X_clip_bool_2*Y_clip_bool_1*Y_clip_bool_2 == True)[0]
    ClippedInvisIndex = np.where(X_clip_bool_1*X_clip_bool_2*Y_clip_bool_1*Y_clip_bool_2 == False)[0]
    
#     XYZ_ndc = np.copy(XYZ_clip_h[ClippedVisIndex][:,0:3])/XYZ_clip_h[ClippedVisIndex][:,3].reshape(-1,1)
    XYZ_ndc = np.copy(XYZ_clip_h[:,0:3])/XYZ_clip_h[:,3].reshape(-1,1)
    
    Near = (ImageWidth//2)/np.tan((FOV/2)*(pi/180.))
    Far = 10*Near
    x_origin = 0
    y_origin = 0

    XYZ_window = np.zeros_like(XYZ_ndc)
    XYZ_window[:,0] = (ImageWidth/2)*XYZ_ndc[:,0] + (x_origin+(ImageWidth/2))
    XYZ_window[:,1] = (ImageHeight/2)*XYZ_ndc[:,1] + (y_origin+(ImageHeight/2))
    XYZ_window[:,2] = ((Far-Near)/2)*XYZ_ndc[:,2] + ((Far+Near)/2)

#     InFront_of_Cam_point_indx = np.where(XYZ_ndc[:,2] <= ((Far+Near)/(Far-Near)))[0]
#     VisiblePoints = XYZ_window[InFront_of_Cam_point_indx]

    return XYZ_window, ClippedVisIndex, ClippedInvisIndex


def TorchBackProjection(image_uv_indices_, image_depth_O3D,
                        cam_x_, cam_y_, cam_z_,
                        theta1_, theta2_, theta3_,
                        ViewAngle_):
    
    width_img = image_depth_O3D.shape[1]
    height_img = image_depth_O3D.shape[0]

    uv_indices = image_uv_indices_.round().to(torch.int64).numpy()
    depthval = torch.tensor(image_depth_O3D[uv_indices[:, 1], uv_indices[:,0]].reshape(-1, 1), dtype=torch.float64)
    
    # image_uv_indices_ may be non-integer
    uv_indices_augmentedwithOnes = torch.tensor(np.hstack([image_uv_indices_, np.ones((uv_indices.shape[0], 1))]), dtype=torch.float64)

    N_p = image_uv_indices_.shape[0]
    Y_invert = torch.tensor([1, -1, 1], dtype=torch.float64).repeat(N_p,1)
    Y_invert_height = torch.tensor([0, height_img, 0]).repeat(N_p,1)

    uv_back = ((uv_indices_augmentedwithOnes - Y_invert_height)*Y_invert)
    uv_back_depth = uv_back*(-1*depthval)

    focal_length = - (height_img)/(2*torch.tan((ViewAngle_ /2)*(pi/180.)))

    camera_matrix = torch.tensor([[focal_length, 0, width_img/2-0.5],
                                  [0, focal_length, height_img/2-0.5],
                                  [0, 0, 1]], dtype=torch.float64)

    back2 = torch.bmm(torch.linalg.inv(camera_matrix).repeat(N_p,1).reshape(N_p,3,3), uv_back_depth.unsqueeze(2))

    THETA = torch.stack((theta1_, theta2_, theta3_), -1)
    THETA_rad = THETA*(pi/180)
    RotMatrix = TorchEulerAnglesToRotationMatrixDegree(THETA_rad)
    back3 = torch.bmm(torch.linalg.inv(RotMatrix).repeat(N_p,1).reshape(N_p,3,3), back2).squeeze(2)

    Cam_pos = torch.stack((cam_x_, cam_y_, cam_z_), -1)
    tvec = - Cam_pos
    xyz_backproj = (back3 - tvec)

    return xyz_backproj


def show_mask(mask, ax, alpha, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([255/255, 0/255, 0/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def GeoXYZ_to_RasterPixel(XYZ, GDAL_GeoTransform):
    g0 = GDAL_GeoTransform[0]
    g1 = GDAL_GeoTransform[1]
    g2 = GDAL_GeoTransform[2]
    g3 = GDAL_GeoTransform[3]
    g4 = GDAL_GeoTransform[4]
    g5 = GDAL_GeoTransform[5]

    X = XYZ[:,0]
    Y = XYZ[:,1]
    
    if g4 != 0:
        coeff = g1/g4
        j_vertical = ((X - coeff*Y) - (g0 - g3*coeff))/(g2 - g5*coeff)
        i_horizental = (X - j_vertical*g2 - g0)/g1
    else:
        i_horizental = (X -g0)/g1
        j_vertical = (Y -g3)/g5


    j_i = np.hstack([j_vertical.reshape(-1,1), i_horizental.reshape(-1,1)]).round().astype('int64')
    j_i_unique = np.unique(j_i, axis=0)
    return j_i


def convert2DarrayToImageRGB(ary, colorMap='viridis'):
    vmin, vmax = np.nanmin(ary), np.nanmax(ary)
    normalized_ary = (ary - vmin) / (vmax - vmin)
    colormap = matplotlib.colormaps[colorMap] 
    ary_rgb = colormap(normalized_ary)[:, :, :3]
    nan_mask = np.isnan(normalized_ary)
    ary_rgb[nan_mask] = [1, 1, 1]  # White color
    ary_rgb = (np.array(ary_rgb)*255).round().astype('uint8')
    return ary_rgb


