import numpy as np
import cupy as cp
from einops import rearrange, reduce

from typing import Union, Dict

try:
  cp_array_class = cp.core.core.ndarray
except:
  cp_array_class = cp._core.core.ndarray
Array = Union[np.ndarray, cp_array_class]


class FruitFlyNet():
  """Fruit fly network as described in "Can a Fruit Fly Learn Word Embeddings?"
  (arXiv:2101.06887)

  Args:
    input_dim: number of input features
    output_dim: number of output features
    k: number of top output activations to keep
    lr: learning rate
  """

  def __init__(self, input_dim: int, output_dim: int,  k: int, lr: int) -> None:
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.k = k
    self.lr = lr
    self.weights = np.random.rand(output_dim, input_dim)
    self.training = True
    self.xp = np

  def __call__(self, x: Array, probs: Array) -> None:
    """Creates bio-hash. If `self.training` is True, updates weights after
    creating bio-hash.
    
    Args:
      x: input of shape batch x input_features
      probs: probabilities of each element in input; has shape 
             batch x input_features, where each row should sum up to 1
    """

    b = x.shape[0]
    activations = self.xp.inner(self.weights, x)
    out = self.xp.zeros_like(activations)
    # self.xp.put_along_axis(out, activations.argsort(axis=0)[-self.k:], 1, axis=0)
    out[rearrange(activations.argsort(axis=0)[-self.k:], 'k b -> (b k)'),
        rearrange(self.xp.indices((self.k, b))[1], 'k b -> (b k)')] = 1

    if self.training:
      self._backward(x, probs, activations)

    return rearrange(out, 'd b -> b d')

  def _backward(self, x: Array, probs: Array, activations: Array) -> None:
    """Updates weights
    
    Args:
      x: input of shape batch x input_features
      probs: probabilities of each element in input; has shape 
             batch x input_features, where each row should sum up to 1
      activations: output activations of shape output_features x batch
    """

    assert self.training, "Cannot update weights in eval mode"
    normalized_x = rearrange(x / probs, 'b d -> b () d')
    activations = rearrange(activations == activations.max(axis=0), 'd b -> b d ()')
    # activations = rearrange(activations == reduce(activations, 'd b -> b', 'max'), 'd b -> b d ()')
    normalized_weights = rearrange(self.xp.inner(self.weights, normalized_x), 'd b () -> b d ()')
    self.weights += self.lr * (activations * (normalized_x - normalized_weights * self.weights)).sum(axis=0)
    # self.weights += reduce(activations * (normalized_x - normalized_weights * self.weights), 'b o i -> o i', 'sum')
  
  def state_dict(self) -> Dict[str, Array]:
    """Returns dictionary of key "weights" and the weight array as the value"""

    return {'weights': cp.asnumpy(self.weights).copy()}

  def load_state_dict(self, state_dict: Dict[str, Array]) -> None:
    """Loads weights
    
    Args:
      state_dict: dictionary with key "weights" and weight array as the value
    """

    curr_shape, new_shape = self.weights.shape, state_dict['weights'].shape
    assert curr_shape == new_shape, f"Incorrect size for `weights`. Expected {curr_shape}, got {new_shape}."
    self.weights = state_dict['weights']
    self.to('cpu' if self.xp==np else 'gpu')

  def eval(self) -> None:
    """Turns off training mode"""

    self.training = False

  def train(self) -> None:
    """Turns on training mode"""

    self.training = True

  def to(self, device: str) -> None:
    """Moves weight array to device
    
    Args:
      device: device to move weights to; must be "cpu" or "gpu"
    """
    
    if device == 'cpu':
      self.weights = cp.asnumpy(self.weights)
    elif device == 'gpu':
      self.weights = cp.asarray(self.weights)
    else:
      raise ValueError("'device' must be either 'cpu' or 'gpu'")
    self.xp = cp.get_array_module(self.weights)


def bio_hash_loss(weights: Array, x: Array, probs: Array) -> Array:
  """Calculates bio-hash loss from "Bio-Inspired Hashing for Unsupervised 
  Similarity Search"
  (arXiv:2001.04907)

  Args:
    weights: model weights of shape output_features x input_features
    x: input of shape batch x input_features
    probs: probabilities of each element in input; has shape 
           batch x input_features, where each row should sum up to 1

  Returns:
    Array of energy/bio-hash loss for each input vector in batch
  """
  
  xp = cp.get_array_module(weights)
  max_activation_indices = xp.inner(weights, x).argmax(axis=0)
  max_activation_weights = weights[max_activation_indices]
  energy = -xp.inner(max_activation_weights, (x / probs)).diagonal()/xp.sqrt(xp.inner(max_activation_weights, max_activation_weights).diagonal())
  return energy.sum()
