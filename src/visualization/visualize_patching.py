import pdb
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.utils import make_grid
from PIL import Image
import torch

im = Image.open('red_elephants.png')
img = F.pil_to_tensor(im).permute(1, 2, 0) # convert to tensor and reshape to [H x W x 3]

H, W, n_channels = img.size(0), img.size(1), img.size(2)

patch_size = (128, 192)
n_h = int(H / patch_size[0])
n_w = int(W / patch_size[1])
n_patches = int(n_h * n_w)


# reshape data to patches.
# ### Dimension mapping: [n_images x H x W x C] --> [n_images x n_patches x patch_h x patch_w x C]
img_patch = img.reshape(shape=(n_h, patch_size[0], n_w, patch_size[1], n_channels))
img_patch = img_patch.permute(0, 2, 1, 3, 4).reshape(n_patches, patch_size[0], patch_size[1], n_channels)

# img_patch_grid = make_grid(img_patch.permute(0, 3, 1, 2))
# img_patch_grid = F.to_pil_image(img_patch_grid)
# img_patch_grid.show()

# pdb.set_trace()

sub_patch = img_patch[25]#.permute(2, 0, 1)

sub_flat_psh = 1
sub_flat_psw = 1

nh_sub = int(sub_patch.size(0) / sub_flat_psh)
nw_sub = int(sub_patch.size(1) / sub_flat_psw)
np_sub = int(nh_sub * nw_sub)


sub_flat = sub_patch.reshape(nh_sub, sub_flat_psh, nw_sub, sub_flat_psw, n_channels)\
                    .permute(0, 2, 1, 3, 4)\
                    .reshape(np_sub, sub_flat_psh, sub_flat_psw, n_channels)
pdb.set_trace()
F.to_pil_image(make_grid(sub_flat.permute(0, 3, 1, 2), nrow=nw_sub*nh_sub)).show()

# sub_patch

# # reshape [n_images x n_patches x patch_h x patch_w x C] --> [n_images x n_paches x patch_h * patch_w * C]
# img_flat = img_patch.reshape(n_patches, -1)

# X_test = X_test.reshape(X_test.size(0), n_h, patch_size[0], n_w, patch_size[1], n_channels)
# X_test = X_test.permute(0, 1, 3, 2, 4, 5).reshape(X_test.size(0), n_patches, patch_size[0], patch_size[1], n_channels)

# # patch_sanity_check(X_train, img_size, patch_size, n_channels)

# # reshape [n_images x n_patches x patch_h x patch_w x C] --> [n_images x n_paches x patch_h * patch_w * C]
# X_train = X_train.reshape(X_train.size(0), n_patches, -1)
# X_test = X_test.reshape(X_test.size(0), n_patches, -1)