import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {}
itos = {}
for i, x in enumerate(chars):
  stoi[x] = i
  itos[i] = x

def encode(s):
  return [stoi[a] for a in s]

def decode(i):
  return ''.join([itos[x] for x in i])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(low=0,high=len(data) - block_size, size=(batch_size,))
  # inputs
  x = torch.stack([data[i:i + block_size] for i in ix]) # stack of inputs
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix]) # corresponding outputs, one token offset
  return x,y

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


