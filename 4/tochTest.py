import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# The ultimate test:
test_tensor = torch.randn(1, 3, 32, 32).to('cuda')
conv = torch.nn.Conv2d(3, 6, 5).to('cuda')
output = conv(test_tensor)
print("Success! The kernel is working.")