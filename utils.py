import numpy as np
from PIL import Image


def get_pad_size(img_size, patch_size:int, stride:int):
    pad_size = (stride - ((img_size - (patch_size - stride)) % stride)) % stride
    return pad_size


def crop_to_patch(imgs, patch_size:tuple, stride:tuple):
    
    arr = np.array(imgs)
    h, w, c = arr.shape
    ph, pw = patch_size
    sh, sw = int(stride[0]*ph), int(stride[1]*pw)

    pad_h = get_pad_size(h, ph, sh)
    pad_w = get_pad_size(w, pw, sw)
    
    h_num = int(np.ceil((h - (ph-sh))/sh))
    w_num = int(np.ceil((w - (pw-sw))/sw))
    
    patches = []
    for row in range(h_num):
        for col in range(w_num):
            patch = arr[row*sh:row*sh+ph, col*sw:col*sw+pw]
            if patch.shape[:2] != (ph, pw):
                res_h = 0
                res_w = 0
                if patch.shape[0] != ph:
                    res_h = pad_h
                if patch.shape[1] != pw:
                    res_w = pad_w
                patch = arr[row*sh-res_h:row*sh+ph-res_h, col*sw-res_w:col*sw+pw-res_w]

            patches.append( patch )

    patches = np.stack(patches, axis=0)
    return patches


def merge_patches(patches, original_shape, patch_size:tuple, stride:tuple, channel_first:bool=True):
    if channel_first:
        # patches shape : (n, c, h, w) -> (n, h, w, c)
        patches = np.swapaxes(patches, 1, 2)
        patches = np.swapaxes(patches, 2, 3)
        
    h_org, w_org = original_shape[:2]
    c_org = patches.shape[-1]
    
    ph, pw = patch_size
    sh, sw = int(stride[0]*ph), int(stride[1]*pw)

    pad_h = get_pad_size(h_org, ph, sh)
    pad_w = get_pad_size(w_org, pw, sw)
    
    # suakit padding
    img = np.zeros((h_org, w_org, c_org))
    counter = np.zeros((h_org, w_org))
    h, w, c = img.shape

    h_num = int(np.ceil((h - (ph-sh))/sh))
    w_num = int(np.ceil((w - (pw-sw))/sw))

    for row in range(h_num):
        for col in range(w_num):
            ih, iw = img[row*sh:row*sh+ph, col*sh:col*sh+ph].shape[:2]
            res_h = 0
            res_w = 0
            if ih != ph:
                res_h = pad_h
            if iw != pw:
                res_w = pad_w
                    
            img[row*sh-res_h:row*sh+ph-res_h, col*sh-res_w:col*sh+ph-res_w] += patches[row*w_num+col]
            counter[row*sw-res_h:row*sw+pw-res_h, col*sw-res_w:col*sw+pw-res_w] += 1.0

    img /= counter[..., np.newaxis]

    return img[:h_org, :w_org]
