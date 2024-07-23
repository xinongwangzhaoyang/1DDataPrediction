import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = './../data.xlsx'
df = pd.read_excel(file_path)

# 提取目标值和光谱数据
y = df['SOM'].values
X = df.drop(columns=['SOM','name']).values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 创建自定义数据集类
class SOMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = SOMDataset(X_train, y_train)
test_dataset = SOMDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 创建三层1D卷积神经网络模型
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 使用一个dummy的输入来计算展平后的大小
        dummy_input = torch.zeros(1, 1, X_train.shape[1])
        dummy_output = self.forward_conv(dummy_input)
        self.flat_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward_conv(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool1d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool1d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv3(x))
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN1D()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    if(epoch % 10 == 0):
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 进行预测
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        y_pred.extend(outputs.squeeze().tolist())
        y_true.extend(labels.tolist())

# 计算R²和RMSE
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f'R²: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')

# 绘制实际值与预测值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='b', s=50)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)
plt.xlabel('Actual SOM')
plt.ylabel('Predicted SOM')
plt.title('Actual vs Predicted SOM')
plt.show()

# 绘制训练过程中的损失变化
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()
