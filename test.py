import torch

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create two tensors with int32 values
a = torch.diag(torch.ones(21 - 1, dtype=torch.int32), 1).to(device)
b = torch.diag(torch.ones(21, dtype=torch.int32), 0).to(device)

# Perform bitwise multiplication
result = torch.bitwise_and(a, b)

# Move the result tensor back to the CPU if necessary
result = result.cpu()

print("Tensor 1:", a)
print("Tensor 2:", b)
print("Result:", result)