import torchvision
import matplotlib.pyplot as plt

def plot_img_grid(img_tensor, path):
    # img_tensor: (B, C, H, W)
    print(img_tensor.shape)
    grid_img = torchvision.utils.make_grid(img_tensor, nrow=4)
    p = plt.imshow(grid_img.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, bbox_inches='tight')