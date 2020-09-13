import cv2
import numpy as np
import torch
# rendering components
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
)
# datastructures
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene

drawing = False  # true if mouse is pressed


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


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global drawing
    thickness = 20
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(canvas, (x, y), thickness, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


if __name__ == "__main__":
    # Parameters
    N_mono_eyed = 20 # Number of rendered objects
    image_size = 224 # Image size for renderer

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # Prepare meshes
    # Create one eyed in random position
    offsets = np.random.uniform(-0.8, 0.8, size=(N_mono_eyed, 3))
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

    # Optimization parameters
    N = len(meshes[0].verts_packed())
    offset_values = torch.zeros((len(meshes), 2), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([offset_values], lr=0.05)

    # Prepare canvas for hand drawing
    img_size = raster_settings.image_size
    canvas = np.zeros((img_size, img_size, 3), np.uint8)
    cv2.namedWindow('canvas')
    cv2.setMouseCallback('canvas', draw_circle)

    # Main loop
    need_update = False
    is_optimizing = False
    while True:
        # Drawing
        cv2.imshow('canvas', canvas)

        # Update reference image
        if need_update:
            # Reference image
            target = np.array(canvas)
            image_ref = torch.from_numpy(target.max(axis=-1).astype('float32') / 255.)[None, ::]
            image_ref = image_ref.to(device)
            canvas = np.zeros((img_size, img_size, 3), np.uint8)
            need_update = False
            is_optimizing = True

        # Optimizing
        if is_optimizing:
            optimizer.zero_grad()
            new_meshes = []
            for offset_val, mesh in zip(offset_values, meshes):
                offset = torch.nn.functional.pad(offset_val.expand((N, 2)), (0, 1))
                new_meshes.append(mesh.offset_verts(offset))
            new_mesh = join_meshes_as_scene(new_meshes)
            images = renderer_silhouette(new_mesh)[..., 3]
            loss = torch.sum((images - image_ref) ** 2)
            loss.backward()
            optimizer.step()

            # renderer for visualization
            rendered_imgs = renderer(new_mesh)[..., :3]
            rendered_img = rendered_imgs.detach().cpu().numpy()[0]
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
            vis = np.hstack([(255 * rendered_img).astype(np.uint8), target])
            cv2.imshow('result', vis)

        # Key input
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord("s"):
            need_update = True
    cv2.destroyAllWindows()
