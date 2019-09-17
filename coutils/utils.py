import torch
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
  """
  Convert a torch tensor into a numpy ndarray for visualization.

  Inputs:
  - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

  Returns:
  - ndarr: A uint8 numpy array of shape (H, W, 3)
  """
  tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
  ndarr = tensor.to('cpu', torch.uint8).numpy()
  return ndarr


def fix_random_seed(seed_no=0):
  """
  Fix random seed to get a deterministic output

  Inputs:
  - seed_no: seed number to be fixed
  """
  torch.manual_seed(seed_no)
  torch.cuda.manual_seed(seed_no)
  random.seed(seed_no)



def visualize_dataset(X_data, y_data, samples_per_class, class_list):
  """
  Make a grid-shape image to plot

  Inputs:
  - X_data: set of [batch, 3, width, height] data
  - y_data: paired label of X_data in [batch] shape
  - samples_per_class: number of samples want to present
  - class_list: list of class names
    e.g.) ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  Outputs:
  - An grid-image that visualize samples_per_class number of samples per class
  """
  img_half_width = X_data.shape[2] // 2
  samples = []
  for y, cls in enumerate(class_list):
    plt.text(-4, (img_half_width * 2 + 2) * y + (img_half_width + 2), cls, ha='right')
    idxs = (y_data == y).nonzero().view(-1)
    for i in range(samples_per_class):
      idx = idxs[random.randrange(idxs.shape[0])].item()
      samples.append(X_data[idx])

  img = make_grid(samples, nrow=samples_per_class)
  return tensor_to_image(img)
