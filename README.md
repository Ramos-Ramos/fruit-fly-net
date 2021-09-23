# Fruit-Fly-Net 🍄

Unofficial Python implementation of [Can a Fruit Fly Learn Word Embeddings?](https://arxiv.org/abs/2101.06887.pdf) with a PyTorch flavored API.

## Installation

```
pip install git+https://github.com/Ramos-Ramos/fruit-fly-net
```

## Demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ramos-Ramos/fruit-fly-net/blob/main/demo/Fruit_Fly_Net_demo_🍄.ipynb)

Check out our Colab demo!

## Usage

```python
import numpy as xp
from scipy.special import softmax
from fruit_fly_net import FruitFlyNet

model = FruitFlyNet(
  input_dim=40000,  # input dimension size (vocab_size * 2)
  output_dim=600,   # output dimension size
  k=16,             # top k cells to be left active in output layer
  lr=1e-4           # learning rate (learning is performed internally)
)
x = xp.concatenate([xp.argsort(xp.random.rand(2000, 20000)) < i for i in (15, 1)], axis=1)
probs = xp.tile(softmax(xp.random.rand(20000)), 2)
output = model(x, probs)
```

Learning is performed internally as long as the model is in train mode. No need to call `.backward()` or instantiate optimizers. To set the mode, use `.train()` and `.eval()`.

```python
model.train() # will update weights on forward pass
model.eval()  # will not update weights on forward pass
```

To get the loss, use `bio_hash_loss`.

```python
from FruitFlyNet import bio_hash_loss

loss = bio_hash_loss(model.weights, x, probs)
```

To enable gpu learning, move the model to the gpu via `.to` and use cupy instead of numpy.

```python
import cupy as xp

model = FruitFlyNet(
  input_size=40000,
  output_size=600,
  k=16,
  lr=1e-4
)
model.to('gpu')
```

<!---Training and validation loop setups, adapted from [PyTorch's tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

```python
from fruit_fly_net import FruitFlyNet, bio_hash_loss

model = FruitFlyNet(
  input_size=40000,
  output_size=600,
  k=16,
  lr=1e-4
)
x = xp.concatenate([xp.argsort(xp.random.rand(2, 20000)) < i for i in (16, 1)], axis=1)
probs = xp.tile(xp.random.rand(20000), 2)
assert model.training, "Cannot update weights in eval mode"

for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # forward
        _ = model(inputs)
        loss = bio_hash_loss(model.weights, x, probs)

        # print statistics
        running_loss += loss
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

test_x = xp.concatenate([xp.argsort(xp.random.rand(2, 20000)) < i for i in (16, 1)], axis=1)
model.eval()
assert not model.training, "Model weights will update if not in eval mode"

validation_loss = 0.0
for i, data in enumerate(trainloader, 0):
    # forward
    _ = model(inputs)
    validation_loss += bio_hash_loss(model.weights, x, probs)

# print statistics
print('validation_loss: %.3f' % (running_loss / 2000))
```--->

## Citation
```bibtex
@misc{liang2021fruit,
      title={Can a Fruit Fly Learn Word Embeddings?}, 
      author={Yuchen Liang and Chaitanya K. Ryali and Benjamin Hoover and Leopold Grinberg and Saket Navlakha and Mohammed J. Zaki and Dmitry Krotov},
      year={2021},
      eprint={2101.06887},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
