import torch
import torch.nn as nn 
#import torch.nn.functional as F

# --- Define CNN Model Class ---
class EmotionClassifierCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifierCNN, self).__init__()
        
        # 1st block
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)  # to deal with overfitting problem
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)          # to deal with overfitting problem
        
        # 2nd block
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)  # to deal with overfitting problem
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)          # to deal with overfitting problem
        
        # 3rd block
        self.conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)  # to deal with overfitting problem
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.25)          # to deal with overfitting problem
        
        # Flattening
        self.flatten = nn.Flatten()
        
        # Fully Connected layer
        self.fc1 = nn.Linear(in_features=128*6*6, out_features=256)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=0.5)
        
        # Output layer
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)       # Convolution
        x = self.bn1(x)         # Batch normalization
        x = self.relu1(x)       # Activation Function
        x = self.pool1(x)       # Pooling
        x = self.dropout1(x)    # Dropout

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)

        # 최종 완전 연결 레이어 (분류 결과)
        # 분류 문제에서는 마지막 레이어에 활성화 함수를 직접 적용하지 않는 경우가 많습니다.
        # 왜냐하면 손실 함수(nn.CrossEntropyLoss)가 내부적으로 softmax를 포함하고 있기 때문입니다.
        x = self.fc2(x)

        return x
    
# Model instance check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} will be used")

model = EmotionClassifierCNN(num_classes=7)
model.to(device)

print("\n--- Model Architecture ---")
print(model)

print("\n--- Model output check with dummy input ---")
dummy_input = torch.randn(1, 1, 48, 48).to(device) 
output = model(dummy_input)
print(f"logits: {output.shape}")    # (batch_size, num_classes)

print("\nCNN model has been designed successfully.")