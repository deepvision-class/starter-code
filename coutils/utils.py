import torch


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
