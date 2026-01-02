import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os  # 用于创建文件夹和保存图片

# 强制使用能弹窗的后端（Windows保险）
import matplotlib
matplotlib.use('TkAgg')

plt.ion()  # 交互模式

# 超参数
batch_size = 64
lr = 0.0002
z_dim = 100
epochs = 50  # 可改成80、100，CPU也完全扛得住

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载MNIST数据集
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("数据集加载完成！开始训练GAN...")

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 32*32),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, 32, 32)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# 模型初始化
device = torch.device("cpu")
G = Generator().to(device)
D = Discriminator().to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

print(f"使用设备: {device}")

# 训练循环
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        current_batch_size = real_imgs.size(0)

        valid = torch.ones(current_batch_size, 1).to(device)
        fake = torch.zeros(current_batch_size, 1).to(device)

        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = criterion(D(real_imgs), valid)
        z = torch.randn(current_batch_size, z_dim).to(device)
        fake_imgs = G(z)
        fake_loss = criterion(D(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        g_loss = criterion(D(fake_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        if i % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] D loss: {d_loss:.4f} G loss: {g_loss:.4f}")

    # 每轮生成样本并自动保存，只在最后一轮弹窗暂停
    print(f"=== Epoch {epoch+1} 完成，正在生成样本 ===")
    G.eval()
    with torch.no_grad():
        test_z = torch.randn(16, z_dim).to(device)
        samples = G(test_z).cpu()
        
        # 自动保存图片到 results 文件夹
        os.makedirs("results", exist_ok=True)
        save_path = f"results/epoch_{epoch+1:03d}.png"  # 文件名如 epoch_001.png
        fig, axs = plt.subplots(4, 4, figsize=(6, 6))
        for j in range(16):
            axs[j//4, j%4].imshow(samples[j].squeeze(0), cmap='gray')
            axs[j//4, j%4].axis('off')
        plt.suptitle(f"Generated Fake Digits - Epoch {epoch+1}")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"图片已保存: {save_path}")

        # 只在最后一轮弹窗并暂停
        if epoch + 1 == epochs:
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            for j in range(16):
                axs[j//4, j%4].imshow(samples[j].squeeze(0), cmap='gray')
                axs[j//4, j%4].axis('off')
            plt.suptitle(f"最终生成结果 - Epoch {epochs} (超清晰！)")
            plt.show()  # 默认阻塞，窗口会一直开着
            input("训练完成！按回车退出程序...")
    G.train()

print("所有训练结束！结果保存在 results/ 文件夹中。")