---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
from tqdm.notebook import tqdm
import imageio

# datastructures
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
    BlendParams,
    
)

%matplotlib inline
# %matplotlib notebook 
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")
```

# Moving mesh
## Create mesh

```python
def create_circle(r=1, offset=[0, 0, 0], rgb=[1.0, 1.0, 1.0], device="cpu", num=16):
    # Create vetices and faces
    theta = np.linspace(0, 2 * np.pi, num=num, endpoint=False)
    x_ = r * np.cos(theta)[:, np.newaxis]
    y_ = r * np.sin(theta)[:, np.newaxis]
    z_ = np.zeros_like(x_)
    vertices = np.concatenate((x_, y_, z_), axis=1)
    vertices = np.r_[vertices, np.array([[0, 0, 0]])].astype(np.float32)
    vertices += np.array(offset)
    faces = np.roll(np.arange(2 * num) // 2, -1).reshape(-1, 2)
    faces = np.c_[faces, np.full(num, num)].astype(np.float32)
    vertices = torch.from_numpy(vertices)
    faces = torch.from_numpy(faces)

    # Set color
    verts_rgb = torch.zeros_like(vertices)[None]
    verts_rgb += torch.from_numpy(np.array(rgb, dtype=np.float32))
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Construct mesh
    mesh = Meshes(
        verts=[vertices.to(device)], faces=[faces.to(device)], textures=textures)
    return mesh


def create_mono_eyed(offset=[0, 0, 0], device="cpu"):
    x, y, z = offset

    # Move in front of camera
    z = + 1
    meshes = []

    base_scale = np.random.uniform(0.15, 0.20)
    meshes.append(create_circle(r=base_scale, offset=[x, y, z], rgb=[1, 0, 0], device=device))
    meshes.append(create_circle(r=base_scale * np.random.uniform(0.3, 0.5), offset=[x, y, z - 0.1], rgb=[1, 1, 1], device=device))
    meshes.append(create_circle(r=base_scale * np.random.uniform(0.1, 0.25), offset=[x, y, z - 0.2], rgb=[0, 0, 1], device=device))
    mesh = join_meshes_as_scene(meshes)
    return mesh
```

```python
N_mono_eyed = 16
device = torch.device("cuda:0")
mesh=create_mono_eyed()

offsets = np.random.uniform(-0.8, 0.8, size=(N_mono_eyed, 3))

# Create one eyed in random position
meshes = []
for offset in offsets:
    meshes.append(create_mono_eyed(offset,device))

```

```python
# Renderer setting
cameras = FoVOrthographicCameras(device=device)

lights = PointLights(
    ambient_color=((0.99, 0.99, 0.99),),
    diffuse_color=((0, 0, 0),),
    specular_color=((0, 0, 0),),
    device=device,
)
raster_settings = RasterizationSettings()
blend_params = BlendParams(background_color=(0.0, 0.0, 0.0), sigma=2e-4)
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(
        device=device, cameras=cameras, lights=lights, blend_params=blend_params
    ),
)
soft_raster_settings = RasterizationSettings(
    faces_per_pixel=3, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma
)
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=soft_raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


```

```python
# Define offset (x,y) of meshes
N = len(meshes[0].verts_packed())
offset_values = torch.zeros((len(meshes), 2), device=device)
# Example
offset_values[0, 0] += 2
offset_values[0, 1] += 2

new_meshes = []
for i, mesh in enumerate(meshes):
    offset = torch.nn.functional.pad(offset_values[i].expand((N, 2)), (0, 1))
    new_meshes.append(mesh.offset_verts(offset))
```

```python
fig, ax = plt.subplots(1, 2)

mesh = join_meshes_as_scene(meshes)
images = renderer(mesh)[...,:3]
image = images.detach().cpu().numpy()[0]
ax[0].imshow(image)

new_mesh = join_meshes_as_scene(new_meshes)
images = renderer(new_mesh)[...,:3]
image = images.detach().cpu().numpy()[0]
ax[1].imshow(image)


```

```python
images = renderer_silhouette(new_mesh)[...,3]
image = images.detach().cpu().numpy()[0]
plt.imshow(image)
```

## Optimizing using differential renderer

```python
import cv2
img_size = raster_settings.image_size
target = np.zeros((img_size,img_size,3), dtype=np.uint8)

# Add text
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 46
font_scale = 12
target = cv2.putText(target, 'A', (30, img_size-20), font,  
                   font_scale, (255,0,0), thickness, cv2.LINE_AA) 
plt.imshow(target)
```

```python
def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()
```

```python
# Reference images
image_ref = torch.from_numpy(target.max(axis=-1).astype('float32') / 255.)[None, ::]
image_ref = image_ref.to(device)


# Optimization
N = len(meshes[0].verts_packed())
offset_values = torch.zeros((len(meshes), 2), device=device, requires_grad=True)
# optimizer = torch.optim.SGD([offset_values], lr=2, momentum=0.9)
optimizer = torch.optim.Adam([offset_values], lr=0.05)

loop = tqdm(range(50))
for i in loop:   
    optimizer.zero_grad()
    
    new_meshes = []
    for offset_val, mesh in zip(offset_values, meshes):
        offset = torch.nn.functional.pad(offset_val.expand((N, 2)), (0, 1))
        new_meshes.append(mesh.offset_verts(offset))
        
    new_mesh = join_meshes_as_scene(new_meshes)
#     images = renderer(new_mesh)[...,:3]
    images = renderer_silhouette(new_mesh)[...,3]
    loss = torch.sum((images - image_ref)**2)
    loss.backward()
    optimizer.step()
    
    loop.set_description(f'Optimizing..., loss:{loss:.3f}')

    images = renderer(new_mesh)[...,:3]
    image = images.detach().cpu().numpy()[0]
    
    vis = np.hstack([(255*image).astype(np.uint8), target])
    imsave('/tmp/_tmp_%04d.png' % i, vis)
    if i%5==0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(target)
make_gif("optimization.gif")


```

## Fun application

```python
N_mono_eyed = 20
image_size = 224
device = torch.device("cuda:0")
mesh=create_mono_eyed()

offsets = np.random.uniform(-0.8, 0.8, size=(N_mono_eyed, 3))

# Create one eyed in random position
meshes = []
for offset in offsets:
    meshes.append(create_mono_eyed(offset,device))

    
# Renderer setting
cameras = FoVOrthographicCameras(device=device)

lights = PointLights(
    ambient_color=((0.99, 0.99, 0.99),),
    diffuse_color=((0, 0, 0),),
    specular_color=((0, 0, 0),),
    device=device,
)
raster_settings = RasterizationSettings(image_size=image_size)
blend_params = BlendParams(background_color=(0.0, 0.0, 0.0), sigma=2e-4)
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(
        device=device, cameras=cameras, lights=lights, blend_params=blend_params
    ),
)
soft_raster_settings = RasterizationSettings(
    image_size=image_size,
    faces_per_pixel=3, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma
)
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=soft_raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

# Optimization
N = len(meshes[0].verts_packed())
offset_values = torch.zeros((len(meshes), 2), device=device, requires_grad=True)
# optimizer = torch.optim.SGD([offset_values], lr=2, momentum=0.9)
optimizer = torch.optim.Adam([offset_values], lr=0.05)
```

```python
import cv2
import numpy as np

drawing = False # true if mouse is pressed

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing
    thickness = 20
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(canvas,(x,y),thickness,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

```

```python
img_size = raster_settings.image_size
canvas = np.zeros((img_size,img_size,3), np.uint8)
cv2.namedWindow('canvas')
cv2.setMouseCallback('canvas', draw_circle)

need_update = False
is_optimizing = False
vis_record = []
while(1):
    
    # Drawing 
    cv2.imshow('canvas',canvas)

    # Optimizing
    if need_update:
        # Reference image
        target = np.array(canvas)
        image_ref = torch.from_numpy(target.max(axis=-1).astype('float32') / 255.)[None, ::]
        image_ref = image_ref.to(device)
        canvas = np.zeros((img_size,img_size,3), np.uint8)
        need_update = False
        is_optimizing = True
        
    if is_optimizing:
        optimizer.zero_grad()
        new_meshes = []
        for offset_val, mesh in zip(offset_values, meshes):
            offset = torch.nn.functional.pad(offset_val.expand((N, 2)), (0, 1))
            new_meshes.append(mesh.offset_verts(offset))
        new_mesh = join_meshes_as_scene(new_meshes)
        images = renderer_silhouette(new_mesh)[...,3]
        loss = torch.sum((images - image_ref)**2)
        loss.backward()
        optimizer.step()

        rendered_imgs = renderer(new_mesh)[...,:3]
        rendered_img = rendered_imgs.detach().cpu().numpy()[0]
        rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
        vis = np.hstack([(255*rendered_img).astype(np.uint8), target])
        vis_record.append(vis)
        cv2.imshow('result',vis)
        
    #
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("s"):
        need_update = True
cv2.destroyAllWindows()
```

```python
for i, img in enumerate(vis_record):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imsave('/tmp/_tmp_%04d.png' % i, img)
make_gif("hand_drawing_optim.gif")
```

```python

```

```python

```

# Explore rendering
## Create mesh

```python

import torch
import numpy as np
from pytorch3d.structures import Meshes
# rendering components
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    TexturesVertex,
    BlendParams
)

device = torch.device("cuda:0")

# create two triangle meshes
vertices = torch.from_numpy(
    np.array(
        [[1, 1, 0], [-1, 1, 0], 
         [-1, -1, 0], [1, -1, 0]],
        dtype=np.float32))

faces = torch.from_numpy(np.array([
    [0, 1, 2],
    [0, 2, 3],
]))

#
verts_rgb = torch.ones_like(vertices)[None]
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Construct mesh
mesh = Meshes(
    verts=[vertices.to(device)], faces=[faces.to(device)], textures=textures)
```

```python
default_setting = RasterizationSettings()
default_setting = PointLights()
for it in dir(default_setting):
    if it[0] != '_':
        print(it, getattr(default_setting, it))
```

```python
kEpsilon = 1e-8
R, T = look_at_view_transform(2.0, 0, 0, device=device)
cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
raster_settings = RasterizationSettings(image_size=64, blur_radius=kEpsilon)
blend_params = BlendParams(background_color=(0.,0.,0.), sigma=0.0, gamma=0.0)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )
)
```

## Normal rendering

```python
offset_x = 0.0
offset_y = 0.01
offset = torch.zeros_like(mesh.verts_packed())
offset[:, 0] = offset_x
offset[:, 1] = offset_y
new_mesh = mesh.offset_verts(offset)
```

```python
new_mesh = mesh.offset_verts(offset)
```

```python
fig, ax = plt.subplots(1, 2)
images = renderer(mesh)
image = images.detach().cpu().numpy()[0][...,:3]
ax[0].imshow(image)
images = renderer(new_mesh)
image = images.detach().cpu().numpy()[0][...,:3]
ax[1].imshow(image)

```

```python
images = renderer(mesh)
image = images.detach().cpu().numpy()[0][...,:3]
plt.imsave('gap_between_meshes.png', image)

```

```python

```

```python
# pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists
fragments = renderer.rasterizer(mesh)
```

```python
fig, ax = plt.subplots(2,2, figsize=(6,6))
image = fragments.pix_to_face.squeeze().detach().cpu().numpy()
ax[0][0].imshow(image)
image = fragments.zbuf.squeeze().detach().cpu().numpy()
ax[0][1].imshow(image)
image = fragments.bary_coords.squeeze().detach().cpu().numpy()
ax[1][0].imshow(image)
image = fragments.dists.squeeze().detach().cpu().numpy()
ax[1][1].imshow(image)
```

```python
mesh_on_screen = renderer.rasterizer.transform(mesh)
verts_on_screen = mesh_on_screen.verts_packed().detach().cpu().numpy()
fig, ax = plt.subplots(1, 1)
ax.plot(verts_on_screen[:, 0], verts_on_screen[:, 1], 'ro')
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 1)
ax.set_aspect("equal")
```

## In details

```python
cameras = renderer.rasterizer.cameras
len(mesh)
```

```python
cameras.get_world_to_view_transform().get_matrix()
```

```python
# the same as meshes_screen = renderer.rasterizer.transform(mesh).verts_packed()
verts_world = mesh.verts_padded()
verts_view = cameras.get_world_to_view_transform().transform_points(
    verts_world
)
verts_screen = cameras.get_projection_transform().transform_points(
    verts_view
)
verts_screen[..., 2] = verts_view[..., 2]
meshes_screen = mesh.update_padded(new_verts_padded=verts_screen)
```

```python
raster_settings = renderer.rasterizer.raster_settings
# By default, turn on clip_barycentric_coords if blur_radius > 0.
# When blur_radius > 0, a face can be matched to a pixel that is outside the
# face, resulting in negative barycentric coordinates.
clip_barycentric_coords = raster_settings.clip_barycentric_coords
if clip_barycentric_coords is None:
    clip_barycentric_coords = raster_settings.blur_radius > 0.0
```

```python
from pytorch3d.renderer import rasterize_meshes
from typing import NamedTuple, Optional
# Class to store the outputs of mesh rasterization
class Fragments(NamedTuple):
    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: torch.Tensor


```

```python

def edge_function(p, v0, v1):
    r"""
    Determines whether a point p is on the right side of a 2D line segment
    given by the end points v0, v1.
    Args:
        p: (x, y) Coordinates of a point.
        v0, v1: (x, y) Coordinates of the end points of the edge.
    Returns:
        area: The signed area of the parallelogram given by the vectors
              .. code-block:: python
                  B = p - v0
                  A = v1 - v0
                        v1 ________
                          /\      /
                      A  /  \    /
                        /    \  /
                    v0 /______\/
                          B    p
             The area can also be interpreted as the cross product A x B.
             If the sign of the area is positive, the point p is on the
             right side of the edge. Negative area indicates the point is on
             the left side of the edge. i.e. for an edge v1 - v0
             .. code-block:: python
                             v1
                            /
                           /
                    -     /    +
                         /
                        /
                      v0
    """
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])


def barycentric_coordinates_clip(bary):
    """
    Clip negative barycentric coordinates to 0.0 and renormalize so
    the barycentric coordinates for a point sum to 1. When the blur_radius
    is greater than 0, a face will still be recorded as overlapping a pixel
    if the pixel is outisde the face. In this case at least one of the
    barycentric coordinates for the pixel relative to the face will be negative.
    Clipping will ensure that the texture and z buffer are interpolated correctly.
    Args:
        bary: tuple of barycentric coordinates
    Returns
        bary_clip: (w0, w1, w2) barycentric coordinates with no negative values.
    """
    # Only negative values are clamped to 0.0.
    w0_clip = torch.clamp(bary[0], min=0.0)
    w1_clip = torch.clamp(bary[1], min=0.0)
    w2_clip = torch.clamp(bary[2], min=0.0)
    bary_sum = torch.clamp(w0_clip + w1_clip + w2_clip, min=1e-5)
    w0_clip = w0_clip / bary_sum
    w1_clip = w1_clip / bary_sum
    w2_clip = w2_clip / bary_sum

    return (w0_clip, w1_clip, w2_clip)


def barycentric_coordinates(p, v0, v1, v2):
    """
    Compute the barycentric coordinates of a point relative to a triangle.
    Args:
        p: Coordinates of a point.
        v0, v1, v2: Coordinates of the triangle vertices.
    Returns
        bary: (w0, w1, w2) barycentric coordinates in the range [0, 1].
    """
    area = edge_function(v2, v0, v1) + kEpsilon  # 2 x face area.
    w0 = edge_function(p, v1, v2) / area
    w1 = edge_function(p, v2, v0) / area
    w2 = edge_function(p, v0, v1) / area
    return (w0, w1, w2)


def point_line_distance(p, v0, v1):
    """
    Return minimum distance between line segment (v1 - v0) and point p.
    Args:
        p: Coordinates of a point.
        v0, v1: Coordinates of the end points of the line segment.
    Returns:
        non-square distance to the boundary of the triangle.
    Consider the line extending the segment - this can be parameterized as
    ``v0 + t (v1 - v0)``.
    First find the projection of point p onto the line. It falls where
    ``t = [(p - v0) . (v1 - v0)] / |v1 - v0|^2``
    where . is the dot product.
    The parameter t is clamped from [0, 1] to handle points outside the
    segment (v1 - v0).
    Once the projection of the point on the segment is known, the distance from
    p to the projection gives the minimum distance to the segment.
    """
    if p.shape != v0.shape != v1.shape:
        raise ValueError("All points must have the same number of coordinates")

    v1v0 = v1 - v0
    l2 = v1v0.dot(v1v0)  # |v1 - v0|^2
    if l2 <= kEpsilon:
        return (p - v1).dot(p - v1)  # v0 == v1

    t = v1v0.dot(p - v0) / l2
    t = torch.clamp(t, min=0.0, max=1.0)
    p_proj = v0 + t * v1v0
    delta_p = p_proj - p
    return delta_p.dot(delta_p)


def point_triangle_distance(p, v0, v1, v2):
    """
    Return shortest distance between a point and a triangle.
    Args:
        p: Coordinates of a point.
        v0, v1, v2: Coordinates of the three triangle vertices.
    Returns:
        shortest absolute distance from the point to the triangle.
    """
    if p.shape != v0.shape != v1.shape != v2.shape:
        raise ValueError("All points must have the same number of coordinates")

    e01_dist = point_line_distance(p, v0, v1)
    e02_dist = point_line_distance(p, v0, v2)
    e12_dist = point_line_distance(p, v1, v2)
    edge_dists_min = torch.min(torch.min(e01_dist, e02_dist), e12_dist)

    return edge_dists_min


def pix_to_ndc(i, S):
    # NDC x-offset + (i * pixel_width + half_pixel_width)
    return -1 + (2 * i + 1.0) / S
```

```python
#TODO(jcjohns): Should we try to set perspective_correct automatically
# based on the type of the camera?
pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
    meshes_screen,
    image_size=raster_settings.image_size,
    blur_radius=raster_settings.blur_radius,
    faces_per_pixel=raster_settings.faces_per_pixel,
    bin_size=raster_settings.bin_size,
    max_faces_per_bin=raster_settings.max_faces_per_bin,
    perspective_correct=raster_settings.perspective_correct,
    clip_barycentric_coords=clip_barycentric_coords,
    cull_backfaces=raster_settings.cull_backfaces,
)

```

```python
# TODO make the epsilon user configurable
kEpsilon = 1e-8
"""
Naive PyTorch implementation of mesh rasterization with the same inputs and
outputs as the rasterize_meshes function.
This function is not optimized and is implemented as a comparison for the
C++/CUDA implementations.
"""
meshes = meshes_screen
N = len(meshes)
# Assume only square images.
# TODO(T52813608) extend support for non-square images.
H, W = raster_settings.image_size, raster_settings.image_size

K = raster_settings.faces_per_pixel
device = meshes.device
blur_radius = raster_settings.blur_radius
cull_backfaces=raster_settings.cull_backfaces
perspective_correct=raster_settings.perspective_correct

verts_packed = meshes.verts_packed()
faces_packed = meshes.faces_packed()
faces_verts = verts_packed[faces_packed]
mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
num_faces_per_mesh = meshes.num_faces_per_mesh()

# Intialize output tensors.
face_idxs = torch.full(
    (N, H, W, K), fill_value=-1, dtype=torch.int64, device=device
)
zbuf = torch.full((N, H, W, K), fill_value=-1, dtype=torch.float32, device=device)
bary_coords = torch.full(
    (N, H, W, K, 3), fill_value=-1, dtype=torch.float32, device=device
)
pix_dists = torch.full(
    (N, H, W, K), fill_value=-1, dtype=torch.float32, device=device
)

# Calculate all face bounding boxes.
# pyre-fixme[16]: `Tuple` has no attribute `values`.
x_mins = torch.min(faces_verts[:, :, 0], dim=1, keepdim=True).values
x_maxs = torch.max(faces_verts[:, :, 0], dim=1, keepdim=True).values
y_mins = torch.min(faces_verts[:, :, 1], dim=1, keepdim=True).values
y_maxs = torch.max(faces_verts[:, :, 1], dim=1, keepdim=True).values
z_mins = torch.min(faces_verts[:, :, 2], dim=1, keepdim=True).values

# Expand by blur radius.
x_mins = x_mins - np.sqrt(blur_radius) - kEpsilon
x_maxs = x_maxs + np.sqrt(blur_radius) + kEpsilon
y_mins = y_mins - np.sqrt(blur_radius) - kEpsilon
y_maxs = y_maxs + np.sqrt(blur_radius) + kEpsilon
```

```python
n = 0
face_start_idx = mesh_to_face_first_idx[n]
face_stop_idx = face_start_idx + num_faces_per_mesh[n]

# Iterate through the horizontal lines of the image from top to bottom.
for yi in range(H):
    # Y coordinate of one end of the image. Reverse the ordering
    # of yi so that +Y is pointing up in the image.
    yfix = H - 1 - yi
    yf = pix_to_ndc(yfix, H)

    # Iterate through pixels on this horizontal line, left to right.
    for xi in range(W):
        # X coordinate of one end of the image. Reverse the ordering
        # of xi so that +X is pointing to the left in the image.
        xfix = W - 1 - xi
        xf = pix_to_ndc(xfix, W)
        top_k_points = []

        # Check whether each face in the mesh affects this pixel.
        for f in range(face_start_idx, face_stop_idx):
            face = faces_verts[f].squeeze()
            v0, v1, v2 = face.unbind(0)

            face_area = edge_function(v0, v1, v2)

            # Ignore triangles facing away from the camera.
            back_face = face_area < 0
            if cull_backfaces and back_face:
                continue

            # Ignore faces which have zero area.
            if face_area == 0.0:
                continue

            outside_bbox = (
                xf < x_mins[f]
                or xf > x_maxs[f]
                or yf < y_mins[f]
                or yf > y_maxs[f]
            )

            # Faces with at least one vertex behind the camera won't
            # render correctly and should be removed or clipped before
            # calling the rasterizer
            if z_mins[f] < kEpsilon:
                continue

            # Check if pixel is outside of face bbox.
            if outside_bbox:
                continue

            # Compute barycentric coordinates and pixel z distance.
            pxy = torch.tensor([xf, yf], dtype=torch.float32, device=device)

            bary = barycentric_coordinates(pxy, v0[:2], v1[:2], v2[:2])
            if perspective_correct:
                z0, z1, z2 = v0[2], v1[2], v2[2]
                l0, l1, l2 = bary[0], bary[1], bary[2]
                top0 = l0 * z1 * z2
                top1 = z0 * l1 * z2
                top2 = z0 * z1 * l2
                bot = top0 + top1 + top2
                bary = torch.stack([top0 / bot, top1 / bot, top2 / bot])

            # Check if inside before clipping
            inside = all(x > 0.0 for x in bary)

            # Barycentric clipping
            if clip_barycentric_coords:
                bary = barycentric_coordinates_clip(bary)
            # use clipped barycentric coords to calculate the z value
            pz = bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2]

            # Check if point is behind the image.
            if pz < 0:
                continue

            # Calculate signed 2D distance from point to face.
            # Points inside the triangle have negative distance.
            dist = point_triangle_distance(pxy, v0[:2], v1[:2], v2[:2])

            signed_dist = dist * -1.0 if inside else dist

            # Add an epsilon to prevent errors when comparing distance
            # to blur radius.
            
            ####### Komatsu modify ########################
            if not inside and (dist-kEpsilon) >= blur_radius:
                continue

            top_k_points.append((pz, f, bary, signed_dist))
            top_k_points.sort()
            if len(top_k_points) > K:
                top_k_points = top_k_points[:K]

        # Save to output tensors.
        for k, (pz, f, bary, dist) in enumerate(top_k_points):
            zbuf[n, yi, xi, k] = pz
            face_idxs[n, yi, xi, k] = f
            bary_coords[n, yi, xi, k, 0] = bary[0]
            bary_coords[n, yi, xi, k, 1] = bary[1]
            bary_coords[n, yi, xi, k, 2] = bary[2]
            pix_dists[n, yi, xi, k] = dist

```

```python
fragments = Fragments(
    pix_to_face=face_idxs, zbuf=zbuf, bary_coords=bary_coords, dists=pix_dists
)
images = renderer.shader(fragments, mesh)
image = images.detach().cpu().numpy()[0][...,:3]
plt.imshow(image)
```

```python
fig, ax = plt.subplots(2,2, figsize=(6,6))
image = fragments.pix_to_face.squeeze().detach().cpu().numpy()
ax[0][0].imshow(image)
image = fragments.zbuf.squeeze().detach().cpu().numpy()
ax[0][1].imshow(image)
image = fragments.bary_coords.squeeze().detach().cpu().numpy()
ax[1][0].imshow(image)
image = fragments.dists.squeeze().detach().cpu().numpy()
ax[1][1].imshow(image)
```

```python
#TODO(jcjohns): Should we try to set perspective_correct automatically
# based on the type of the camera?
pix_to_face, zbuf, bary_coords, dists = rasterize_meshes_python(
    meshes_screen,
    image_size=raster_settings.image_size,
    blur_radius=raster_settings.blur_radius,
    faces_per_pixel=raster_settings.faces_per_pixel,
    bin_size=raster_settings.bin_size,
    max_faces_per_bin=raster_settings.max_faces_per_bin,
    perspective_correct=raster_settings.perspective_correct,
    clip_barycentric_coords=clip_barycentric_coords,
    cull_backfaces=raster_settings.cull_backfaces,
)

```

## Shader in detail

```python
fragments = renderer.rasterizer(mesh)
fig, ax = plt.subplots(2,2, figsize=(6,6))
image = fragments.pix_to_face.squeeze().detach().cpu().numpy()
ax[0][0].imshow(image)
image = fragments.zbuf.squeeze().detach().cpu().numpy()
ax[0][1].imshow(image)
image = fragments.bary_coords.squeeze().detach().cpu().numpy()
ax[1][0].imshow(image)
image = fragments.dists.squeeze().detach().cpu().numpy()
ax[1][1].imshow(image)
```

```python
cameras = renderer.shader.cameras
if cameras is None:
    msg = "Cameras must be specified either at initialization \
        or in the forward pass of HardPhongShader"
    raise ValueError(msg)
```

```python
texels = mesh.textures.sample_textures(
                fragments, faces_packed=mesh.faces_packed()
            )
```

```python
lights = renderer.shader.lights
materials = renderer.shader.materials
blend_params = renderer.shader.blend_params
```

```python
from pytorch3d.ops import interpolate_face_attributes
meshes = mesh
verts = meshes.verts_packed()  # (V, 3)
faces = meshes.faces_packed()  # (F, 3)
vertex_normals = meshes.verts_normals_packed()  # (V, 3)
faces_verts = verts[faces]
faces_normals = vertex_normals[faces]
pixel_coords = interpolate_face_attributes(
    fragments.pix_to_face, fragments.bary_coords, faces_verts
)
pixel_normals = interpolate_face_attributes(
    fragments.pix_to_face, fragments.bary_coords, faces_normals
)

```

```python
image = texels.squeeze().detach().cpu().numpy()
plt.imshow(image)
```

```python
ambient, diffuse, specular = _apply_lighting(
    pixel_coords, pixel_normals, lights, cameras, materials
)
colors = (ambient + diffuse) * texels + specular
```

```python
default_setting = blend_params
for it in dir(default_setting):
    if it[0] != '_':
        print(it, getattr(default_setting, it))
```

```python
colors = phong_shading(
    meshes=meshes,
    fragments=fragments,
    texels=texels,
    lights=lights,
    cameras=cameras,
    materials=materials,
)
images = hard_rgb_blend(colors, fragments, blend_params)
return images
```

```python
mesh.verts_packed()
```

```python


images = renderer.shader(fragments, mesh)
image = images.detach().cpu().numpy()[0][...,:3]
plt.imshow(image)
```

```python
renderer.shader
```

```python

```

```python

```
