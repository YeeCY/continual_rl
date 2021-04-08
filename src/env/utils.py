import torch
import numpy as np

import torch.nn.functional as F
import torchvision.transforms.functional as TF


def rgb_to_hsv(r, g, b):
    """Convert RGB color to HSV color"""
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc - minc) / maxc
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0
    return h, s, v


def do_green_screen(x, bg):
    """Removes green background from observation and replaces with bg; not optimized for speed"""
    assert isinstance(x, np.ndarray) and isinstance(bg, np.ndarray), 'inputs must be numpy arrays'
    assert x.dtype == np.uint8 and bg.dtype == np.uint8, 'inputs must be uint8 arrays'

    # Get image sizes
    x_h, x_w = x.shape[1:]

    # Convert to RGBA images
    im = TF.to_pil_image(torch.ByteTensor(x))
    im = im.convert('RGBA')
    pix = im.load()
    bg = TF.to_pil_image(torch.ByteTensor(bg))
    bg = bg.convert('RGBA')
    bg = bg.load()

    # Replace pixels
    for x in range(x_w):
        for y in range(x_h):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255., g / 255., b / 255.)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = (100, 80, 70)
            max_h, max_s, max_v = (185, 255, 255)
            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = bg[x, y]

    x = np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]

    return x


def interpolate_bg(bg, size: tuple):
    """Interpolate background to size of observation"""
    bg = torch.from_numpy(bg).float().unsqueeze(0) / 255
    bg = F.interpolate(bg, size=size, mode='bilinear', align_corners=False)
    return (bg * 255).byte().squeeze(0).numpy()


def replace_bg(img, seg, bg):
    """Removes original background from observation and replaces with bg using segmentation mask"""
    assert isinstance(img, np.ndarray) and isinstance(seg, np.ndarray) and isinstance(bg, np.ndarray), \
        'inputs must be numpy arrays'
    assert img.dtype == np.uint8 and seg.dtype == np.int32 and bg.dtype == np.uint8, \
        'image and background must be uint8 arrays, and segmentation must be int32 arrays'

    # Get background mask
    # dm_control document about segementation:
    # 	a 2-channel NumPy int32 array of label values where the pixels of each object are labeled with the
    # 	pair (mjModel ID, mjtObj enum object type).
    #   https://github.com/deepmind/dm_control/blob/a669634a9bdd5be5d78654b2370f9ef8fd987817/dm_control/mujoco/engine.py#L192
    #
    bg_mask = np.bitwise_and(seg[0] == -1, seg[0] == -1)

    # Mask original background with the new one
    img[:, bg_mask] = bg[:, bg_mask].copy()

    return img


def round_robin_strategy(num_tasks, last_task=None):
    """A function for sampling tasks in round robin fashion.
    Args:
        num_tasks (int): Total number of tasks.
        last_task (int): Previously sampled task.
    Returns:
        int: task id.
    """
    if last_task is None:
        return 0

    return (last_task + 1) % num_tasks


def uniform_random_strategy(num_tasks, _):
    """A function for sampling tasks uniformly at random.
    Args:
        num_tasks (int): Total number of tasks.
        _ (object): Ignored by this sampling strategy.
    Returns:
        int: task id.
    """
    return random.randint(0, num_tasks - 1)
