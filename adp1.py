import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from torch.fft import rfft2, irfft2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_optimizer import Lookahead

# ----------- مدل FNO -----------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = rfft2(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy, ioxy ->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        x = irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x

class Net2d(nn.Module):
    def __init__(self, width=64, modes1=12, modes2=12):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(2, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = torch.mean(x, dim=(1, 2))
        return x

# داده‌ها: X (ADP)، Y (x, y مختصات نرمال‌شده)، L (x, y مختصات اصلی)
# باید تعریف شده باشند: X_train, Y_train, Y_mean, Y_std, device

# ----------- 3. اضافه کردن AutoEncoder برای Augmentation

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 16 * 16), nn.ReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 2, 3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# آموزش AutoEncoder و ساخت داده افزوده‌شده
print("\nTraining AutoEncoder for data augmentation...")
autoencoder = AutoEncoder().to(device)
ae_opt = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

for epoch in range(20):
    autoencoder.train()
    loss_epoch = 0
    for xb, _ in train_loader:
        xb = xb.to(device)
        ae_opt.zero_grad()
        recon = autoencoder(xb)
        loss = F.mse_loss(recon, xb)
        loss.backward()
        ae_opt.step()
        loss_epoch += loss.item()
    print(f"AE Epoch {epoch+1}: Loss = {loss_epoch / len(train_loader):.6f}")

# ساخت داده افزوده‌شده با نویز کوچک در فضای فشرده
print("\nGenerating augmented data...")
autoencoder.eval()
augmented_x, augmented_y = [], []
with torch.no_grad():
    for i in range(X_train.shape[0]):
        x = X_train[i:i+1].to(device)
        y = Y_train[i:i+1].repeat(10, 1)
        z = autoencoder.encoder(x.repeat(10, 1, 1, 1))
        z_noisy = z + 0.01 * torch.randn_like(z)
        x_aug = autoencoder.decoder(z_noisy)
        augmented_x.append(x_aug.cpu())
        augmented_y.append(y)

augmented_x = torch.cat(augmented_x)
augmented_y = torch.cat(augmented_y)

X_train_aug = torch.cat([X_train, augmented_x], dim=0)
Y_train_aug = torch.cat([Y_train, augmented_y], dim=0)

train_loader = DataLoader(TensorDataset(X_train_aug, Y_train_aug), batch_size=64, shuffle=True)

# ----------- 4. آموزش مدل به تفکیک x و y

def hybrid_loss(preds, targets, alpha=0.8):
    mse = F.mse_loss(preds, targets)
    euclidean = torch.norm(preds - targets, dim=1).mean()
    return alpha * mse + (1 - alpha) * euclidean

def train_model(index):
    model = Net2d(width=64).to(device)
    base_optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = Lookahead(base_optimizer)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_loss = float('inf')
    best_model = None

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb[:, index:index+1].to(device)
            optimizer.zero_grad()
            preds = model(xb)
            preds = preds[:, index:index+1]
            loss = F.mse_loss(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(epoch)
        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model

print("\nTraining model for x...")
model_x = train_model(index=0)
print("\nTraining model for y...")
model_y = train_model(index=1)

# ----------- 5. ارزیابی جداگانه روی 1000 داده اصلی
model_x.eval()
model_y.eval()

X_eval = X[:1000]
Y_eval = L[:1000]
eval_loader = DataLoader(TensorDataset(X_eval, Y_eval), batch_size=64)

preds_x_all, preds_y_all, targets_all = [], [], []

with torch.no_grad():
    for xb, yb in eval_loader:
        xb = xb.to(device)
        px = model_x(xb)[:, 0:1]
        py = model_y(xb)[:, 0:1]
        preds_x_all.append(px * Y_std[0] + Y_mean[0])
        preds_y_all.append(py * Y_std[1] + Y_mean[1])
        targets_all.append(yb * Y_std + Y_mean)

preds_x = torch.cat(preds_x_all).cpu().numpy()
preds_y = torch.cat(preds_y_all).cpu().numpy()
preds = np.concatenate([preds_x, preds_y], axis=1)
targets = torch.cat(targets_all).cpu().numpy()

mse_x = mean_squared_error(targets[:, 0], preds[:, 0])
mse_y = mean_squared_error(targets[:, 1], preds[:, 1])
rmse_x = np.sqrt(mse_x)
rmse_y = np.sqrt(mse_y)
euc = np.mean(np.linalg.norm(preds - targets, axis=1))

print(f"\nFinal RMSE (x): {rmse_x:.4f} meters")
print(f"Final RMSE (y): {rmse_y:.4f} meters")
print(f"Mean Euclidean Distance Error: {euc:.4f} meters")

# ----------- 6. مصورسازی
errors = np.linalg.norm(preds - targets, axis=1)
plt.hist(errors, bins=50, color='teal', edgecolor='black')
plt.title("Distribution of Localization Errors")
plt.xlabel("Error (meters)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
