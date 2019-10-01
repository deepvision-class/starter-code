import torch

@torch.no_grad()
def rel_error(pred, true, eps=1e-8):
  """
  Returns relative error: (|pred| - |true|) / |true|

  Inputs:
  - pred: A torch tensor of the prediction of the true tensor
  - true: A torch tensor of the true tensor
  - eps: Small epsilon for numerical stability

  Returns:
  - Relative error between pred and true in floating point
  """
  return ((pred-true).abs() / true.abs().clamp(min=eps)).max().item()


@torch.no_grad()
def compute_numeric_gradient(f, x, dy=None, h=1e-5):
  """
  Compute a numeric gradient of the function f at the point x.
  Inputs:
  - f: The function for which to compute the numeric gradient
  - x: A torch tensor giving the point at which to evaluate the gradient
  - dy: (Optional) A torch tensor giving the upstream gradient dL/dy. If this
    is not provided then initialize it to be all ones.
  - h: (Optional) A Python float giving the step size to use 
  
  Returns:
  - dx: A torch tensor giving the downstream gradient dL/dx
  """
  # First run the function unmodified
  y = f(x)

  # Initialize upstream gradient to all ones if not provided
  if dy is None:
    dy = torch.ones_like(y)

  # Initialize downstream gradient to zeros
  dx = torch.zeros_like(x)

  # Get flattened views of everything
  x_flat = x.contiguous().view(-1)
  y_flat = y.contiguous().view(-1)
  dx_flat = dx.contiguous().view(-1)
  dy_flat = dy.contiguous().view(-1)
  for i in range(dx_flat.shape[0]):
    # Compute numeric derivative dy/dx[i]
    orig = x_flat[i].item()
    x_flat[i] = orig + h
    yph = f(x).clone().view(-1)
    x_flat[i] = orig - h
    ymh = f(x).clone().view(-1)
    x_flat[i] = orig
    dy_dxi = (yph - ymh) / (2.0 * h) 

    # Use chain rule to compute dL/dx[i] =  dL/dy . dy/dx[i]
    dx_flat[i] = dy_flat.dot(dy_dxi).item()
  
  dx = dx_flat.view(x.shape)
  return dx

