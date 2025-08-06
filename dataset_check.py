import os
import matplotlib.pyplot as plt
from PIL import Image # Pillow 라이브러리, 이미지 로딩에 사용

# 데이터셋의 기본 경로 설정
data_dir = './fer2013/'

# 감정 클래스 정의 (FER-2013 데이터셋의 레이블)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("--- 데이터셋 폴더 구조 및 개수 확인 ---")

# 훈련(train) 데이터셋 확인
print("\n[Train Dataset]")
train_path = os.path.join(data_dir, 'train')
train_counts = {}
for emotion in emotions:
    emotion_path = os.path.join(train_path, emotion)
    count = len(os.listdir(emotion_path))
    train_counts[emotion] = count
    print(f"  {emotion}: {count} images")

print(f"총 훈련 이미지 수: {sum(train_counts.values())}")

# 테스트(test) 데이터셋 확인
print("\n[Test Dataset]")
test_path = os.path.join(data_dir, 'test')
test_counts = {}
for emotion in emotions:
    emotion_path = os.path.join(test_path, emotion)
    count = len(os.listdir(emotion_path))
    test_counts[emotion] = count
    print(f"  {emotion}: {count} images")

print(f"총 테스트 이미지 수: {sum(test_counts.values())}")

# 데이터 분포 시각화 (선택 사항)
# 훈련 데이터셋의 감정 분포 막대 그래프
plt.figure(figsize=(10, 5))
plt.bar(train_counts.keys(), train_counts.values(), color='skyblue')
plt.title('Emotion Distribution in Training Dataset')
plt.xlabel('Emotion')
plt.ylabel('Number of Images')
plt.show()

# 이미지 샘플 몇 개 확인 (선택 사항)
print("\n--- 이미지 샘플 확인 ---")
sample_image_path = os.path.join(train_path, 'happy', os.listdir(os.path.join(train_path, 'happy'))[0])
print(f"샘플 이미지 경로: {sample_image_path}")
try:
    img = Image.open(sample_image_path)
    plt.imshow(img, cmap='gray') # 흑백 이미지이므로 cmap='gray'
    plt.title(f"Sample Image (Happy) - Size: {img.size}")
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print("샘플 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")