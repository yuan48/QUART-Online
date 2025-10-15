import torch

def calculate_mse(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2)

def calculate_psnr(original, reconstructed):
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 假设像素值范围是 [0, 1]
    return 10 * torch.log10(max_pixel**2 / mse)

def calculate_mae(original, reconstructed):
    return torch.mean(torch.abs(original - reconstructed))

def calculate_nrmse(original, reconstructed):
    mse = calculate_mse(original, reconstructed)
    return torch.sqrt(mse) / (torch.max(original) - torch.min(original))

# 示例数据
original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
reconstructed = torch.tensor([[1.0, 2.0, 2.0], [4.0, 5.0, 5.0]])

# 计算指标
mse = calculate_mse(original, reconstructed)
psnr = calculate_psnr(original, reconstructed)
mae = calculate_mae(original, reconstructed)
nrmse = calculate_nrmse(original, reconstructed)

print(f"MSE: {mse.item()}")
print(f"PSNR: {psnr.item()}")
print(f"MAE: {mae.item()}")
print(f"NRMSE: {nrmse.item()}")