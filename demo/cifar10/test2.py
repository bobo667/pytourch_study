import torch

output = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

print(output.argmax(1))
targets = torch.tensor([[1, 2]])
print()
