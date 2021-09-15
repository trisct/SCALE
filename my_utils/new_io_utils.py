import numpy as np
import mcubes
import trimesh
from plyfile import PlyData, PlyElement
import cv2
import av

def load_packed(filename):
    data = np.load(filename)
    data_dict = {}
    for key in data.files:
        data_dict[key] = data[key]
    
    for key in data_dict.keys():
        print(f"{key}: {type(data_dict[key])}")
        if key != 'scan_name':
            print(f"    shape: {data_dict[key].shape}")
        else:
            print(f"    {data_dict[key]}")

    return data_dict

def wrap_grid_np_batch_last_(grid):
    """
    grid: [res, res, res, ...],
    """
    res = grid.shape[0]
    assert res // 2 * 2 == res
    
    tmp = grid[res//2:].copy()
    grid[res//2:], grid[:res//2] = grid[:res//2], tmp

    tmp = grid[:, res//2:].copy()
    grid[:, res//2:], grid[:, :res//2] = grid[:, :res//2], tmp

    tmp = grid[:, :, res//2:].copy()
    grid[:, :, res//2:], grid[:, :, :res//2] = grid[:, :, :res//2], tmp

def wrap_grid_np_batch_first_(grid):
    """
    grid: [..., res, res, res], 
    """
    res = grid.shape[1]
    assert res // 2 * 2 == res
    
    tmp = grid[..., res//2:].copy()
    grid[..., res//2:], grid[..., :res//2] = grid[..., :res//2], tmp

    tmp = grid[..., res//2:, :].copy()
    grid[..., res//2:, :], grid[..., :res//2, :] = grid[..., :res//2, :], tmp

    tmp = grid[..., res//2:, :, :].copy()
    grid[..., res//2:, :, :], grid[..., :res//2, :, :] = grid[..., :res//2, :, :,], tmp

def wrap_grid_torch_batch_last_(grid):
    """
    grid: [res, res, res, ...], 
    """
    res = grid.shape[0]
    assert res // 2 * 2 == res
    
    tmp = grid[res//2:].clone()
    grid[res//2:], grid[:res//2] = grid[:res//2], tmp

    tmp = grid[:, res//2:].clone()
    grid[:, res//2:], grid[:, :res//2] = grid[:, :res//2], tmp

    tmp = grid[:, :, res//2:].clone()
    grid[:, :, res//2:], grid[:, :, :res//2] = grid[:, :, :res//2], tmp

def wrap_grid_torch_batch_first_(grid):
    """
    grid: [..., res, res, res], 
    """
    res = grid.shape[1]
    assert res // 2 * 2 == res
    
    tmp = grid[..., res//2:].clone()
    grid[..., res//2:], grid[..., :res//2] = grid[..., :res//2], tmp

    tmp = grid[..., res//2:, :].clone()
    grid[..., res//2:, :], grid[..., :res//2, :] = grid[..., :res//2, :], tmp

    tmp = grid[..., res//2:, :, :].clone()
    grid[..., res//2:, :, :], grid[..., :res//2, :, :] = grid[..., :res//2, :, :,], tmp



def load_xyz_and_preprocess(filename, scale_range=0.85, has_normal=False):
    pts = np.loadtxt(filename, dtype=np.float32)
    
    if has_normal:
        pts, nml = pts[:, :3], pts[:, 3:]
    else:
        pts, nml = pts[:, :3], None
    
    scale = 2 * scale_range / (pts.max(0) - pts.min(0)).max()
    trans = (pts.max(0) + pts.min(0)) / 2
    
    pts -= trans[None]
    pts *= scale

    pts_axis_tmp = pts[:, 0].copy()
    pts[:, 0], pts[:, 2] = pts[:, 2], pts_axis_tmp

    return pts, nml, scale, trans

def load_xyz_and_scaletrans(filename, scale, trans, has_normal=False):
    pts = np.loadtxt(filename, dtype=np.float32)
    
    if has_normal:
        pts, nml = pts[:, :3], pts[:, 3:]
    else:
        pts, nml = pts[:, :3], None
        
    pts -= trans[None]
    pts *= scale

    pts_axis_tmp = pts[:, 0].copy()
    pts[:, 0], pts[:, 2] = pts[:, 2], pts_axis_tmp

    return pts, nml

def load_ply_and_preprocess(filename, scale, trans):
    mesh = trimesh.load_mesh(filename)

    mesh.vertices -= trans[None]
    mesh.vertices *= scale
    
    # mesh.show()

    return mesh

def export_obj_from_indicator(filename, idc, iso_val, coord_range, res, scale, trans):
    low, high = coord_range
    d = (high - low) / res
    low, high = low + d/2, high - d/2

    v, f = mcubes.marching_cubes(idc, iso_val)

    scale_grid = (high - low) / (res - 1)
    v *= scale_grid
    v += low

    v /= scale
    v += trans[None]

    mcubes.export_obj(v, f, filename)

def export_scalar_grid_as_ply(filename, qual_field, coord_field, scale, trans, mask=None):
    """
    val_field: [res, res, res]
    coord_field: [res, res, res, 3]
    mask: [res, res, res]
    """

    if mask is not None:
        coord_field = coord_field[mask].reshape(-1, 3)
        #qual_field = qual_field[np.repeat(mask[..., None], 3, 3)]
        qual_field = qual_field[mask]

    vertex_field = np.concatenate([
        coord_field, qual_field[..., None]], -1).reshape(-1, 4)
 
    vertex_plydata = np.empty(
        len(vertex_field),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4')])

    vertex_plydata['x'] = vertex_field[:, 0] / scale + trans[0]
    vertex_plydata['y'] = vertex_field[:, 1] / scale + trans[1]
    vertex_plydata['z'] = vertex_field[:, 2] / scale + trans[2]
    vertex_plydata['quality'] = vertex_field[:, 3]

    vert_elem = PlyElement.describe(vertex_plydata, 'vertex')

    ply_data = PlyData([vert_elem], text=False)
    ply_data.write(filename)

    return ply_data

def export_points_as_xyz(filename, points, normals=None, coord_type='xyz', scale=1., trans=np.zeros(3)):
    if normals is not None:
        pts = np.concatenate([points, normals], 1)
    else:
        pts = points

    assert (coord_type in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']), "coord type must be one of the above"
    
    axes = {}
    axes[coord_type[0]] = pts[:, 0]
    axes[coord_type[1]] = pts[:, 1]
    axes[coord_type[2]] = pts[:, 2]

    pts = np.concatenate([
        np.stack([axes['x'], axes['y'], axes['z']], -1),
        pts[:, 3:]], -1)
    print(pts.shape)
    pts[:, :3] /= scale
    pts[:, :3] += trans[None]
    
    np.savetxt(filename, pts, fmt="%.8f", delimiter=' ')
    
    
def export_grid_as_npy(filename, grid):
    np.save(filename, grid)
    
def export_scalar_img_as_png(filename, img, low=None, high=None, cmap=cv2.COLORMAP_JET):
    """
    grid: [res, res, res] array
    """

    if low is None:
        low = img.min()
    if high is None:
        high = img.max()
    img = np.clip(img, low, high)
    img -=low
    img /= (high - low)
    img *= 255
    img = img.astype(np.uint8)
    img = cv2.applyColorMap(img, cmap)
    
    cv2.imwrite(filename, img)

def export_grid_as_mp4(filename, grid, axis, fps, low=0, high=1, cmap=cv2.COLORMAP_TURBO):
    """
    grid: [res, res, res] array
    """
    assert axis < len(grid.shape)
    if axis != 0:
        grid = np.swapaxes(grid, axis, 0)


    container = av.open(filename, mode='w')
    stream = container.add_stream('mpeg4', rate=fps)
    stream.height = grid.shape[1]
    stream.width = grid.shape[2]
    stream.pix_fmt = 'yuv420p'

    for i in range(grid.shape[0]):
        frame = grid[i]
        frame = np.clip(frame, low, high)
        # frame -= low
        # frame /= (high - low)
        frame *= 255
        frame = frame.astype(np.uint8)
        frame = cv2.applyColorMap(frame, cmap)
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    
    for packet in stream.encode(frame):
            container.mux(packet)
    
    container.close()




    
