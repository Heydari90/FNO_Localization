import sys
sys.path.append('E:/Heydari')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import h5py
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from functools import partial
import matplotlib.pyplot as plt

# ----------- 1. بارگذاری داده‌ها
with h5py.File('ADP_BS1_Nt16_Nc16_BW100MHz_O1_3p5_V1.mat', 'r') as f:
    adp_real = f['ADP']['real'][:]
    adp_imag = f['ADP']['imag'][:]
    locs = f['L'][:]

# بازسازی ADP
adp_complex = adp_real + 1j * adp_imag
adp_complex = adp_complex.reshape(-1, 16, 16)
l_tensor_np = np.array(locs)

# ----------- 2. جداسازی داده‌های آموزش و تست
num_samples = adp_complex.shape[0]
percent = 0.001
num_train = int(num_samples * percent)

perm = np.random.permutation(num_samples)
train_idx = perm[:num_train]
test_idx = perm[num_train:num_train + 1000]

X_complex_train = adp_complex[train_idx]
L_train = l_tensor_np[train_idx]
X_complex_test = adp_complex[test_idx]
L_test = l_tensor_np[test_idx]

# ----------- نرمال‌سازی و تبدیل داده‌ها
def process_data(X_complex, L, X_mean=None, X_std=None, Y_mean=None, Y_std=None, is_train=True):
    X_real = np.real(X_complex)
    X_imag = np.imag(X_complex)
    X_np = np.stack([X_real, X_imag], axis=1)
    X = torch.tensor(X_np, dtype=torch.float32)
    L = torch.tensor(L, dtype=torch.float32)

    if is_train:
        X_mean = X.mean(dim=(0, 2, 3), keepdim=True)
        X_std = X.std(dim=(0, 2, 3), keepdim=True)
        Y_mean = L.mean(dim=0)
        Y_std = L.std(dim=0)
        Y_std[Y_std == 0] = 1.0

    X = (X - X_mean) / X_std
    L = (L - Y_mean) / Y_std
    return X, L, X_mean, X_std, Y_mean, Y_std

X_train, L_train, X_mean, X_std, Y_mean, Y_std = process_data(X_complex_train, L_train, is_train=True)
X_test, L_test, _, _, _, _ = process_data(X_complex_test, L_test, X_mean, X_std, Y_mean, Y_std, is_train=False)

train_loader = DataLoader(TensorDataset(X_train, L_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, L_test), batch_size=64, shuffle=False)

# ----------- 3. تعریف مدل FNO
def compl_mul2d(a, b):
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / np.sqrt(in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho')
        return torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.width = width
        self.fc0 = nn.Conv2d(2, width, 1)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(7)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(7)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(7)])
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(width, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 3)

    def forward(self, x):
        x = self.fc0(x)
        for conv, w, norm in zip(self.convs[:-1], self.ws[:-1], self.norms[:-1]):
            x = F.relu(norm(conv(x) + w(x)))
        x = self.convs[-1](x) + self.ws[-1](x)
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class Net2d(nn.Module):
    def __init__(self, modes1=12, modes2=9, width=64):
        super().__init__()
        self.model = SimpleBlock2d(modes1, modes2, width)
    def forward(self, x):
        return self.model(x)
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ----------- 4. آموزش مدل
def hybrid_loss(preds, targets, alpha=0.8):
    mse = F.mse_loss(preds, targets)
    euclidean = torch.norm(preds - targets, dim=1).mean()
    return alpha * mse + (1 - alpha) * euclidean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net2d().to(device)
print("تعداد پارامترها:", model.count_params())

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_loss = float('inf')
best_model = None

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = hybrid_loss(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model = model.state_dict()

# ----------- 5. ارزیابی نهایی روی ۱۰۰۰ داده تست
model.load_state_dict(best_model)
model.eval()
preds_all, targets_all = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        preds_all.append(preds * Y_std.to(device) + Y_mean.to(device))
        targets_all.append(yb * Y_std.to(device) + Y_mean.to(device))

preds = torch.cat(preds_all, dim=0).cpu().numpy()
targets = torch.cat(targets_all, dim=0).cpu().numpy()

mse = mean_squared_error(targets, preds)
rmse = np.sqrt(mse)
euc = np.mean(np.linalg.norm(preds - targets, axis=1))

print(f"\n📌 Final Evaluation on 1000 Original Test Samples:")
print(f"Final Test MSE: {mse:.6f}")
print(f"Final Test RMSE: {rmse:.6f}")
print(f"Mean Euclidean Distance Error: {euc:.6f} meters")

# ----------- 6. مصورسازی
errors = np.linalg.norm(preds - targets, axis=1)
plt.hist(errors, bins=50, color='teal', edgecolor='black')
plt.title("Distribution of Localization Errors")
plt.xlabel("Error (meters)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
