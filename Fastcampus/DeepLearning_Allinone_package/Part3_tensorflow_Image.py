from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm  # 터미널에서도 가능
from tqdm import tqdm_notebook  # 쥬피터 노트북에서 활용
from tensorflow.keras import layers
import os
from glob import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import zipfile

# -------------------------------------------------------
# 1. 데이터 준비
# -------------------------------------------------------
path_dir = os.getcwd()
data_dir = path_dir + '/Fastcam/data/DeepLearning_Allinone/'

os.listdir(data_dir)

# mnist data zip 파일 압축 해제
mnist_zip = zipfile.ZipFile(data_dir+'mnist_png.zip')
mnist_zip.extractall(data_dir+'mnist')
mnist_zip.close()

os.listdir(data_dir+'mnist/mnist_png/training/0/')[0]
data_train_dir = data_dir + 'mnist/mnist_png/training/'

data_paths = glob(data_train_dir + '*/*.png')
path = data_paths[0]
len(data_paths)

# -------------------------------------------------------
# 2. 이미지 분석
# -------------------------------------------------------
label_nums = os.listdir(data_train_dir)
label_nums

# label 0의 데이터 갯수 확인
len(os.listdir(data_train_dir+'/0'))

# 데이터 별 갯수 비교
nums_dataset = []

for lbl_n in label_nums:
    data_per_class = os.listdir(data_train_dir+lbl_n)
    nums_dataset.append(len(data_per_class))
nums_dataset

plt.bar(list(range(10)), nums_dataset)
plt.title('Number of dataset per class')
plt.show()

# Pillow로 열기
image_pil = Image.open(path)
image_pil
image = np.array(image_pil)
image.shape

plt.imshow(image, 'gray')
plt.show()

# Tensorflow로 열기
gfile = tf.io.read_file(path)
image = tf.io.decode_image(gfile)
image.shape

plt.imshow(image[:, :, 0], 'gray')
plt.show()

# label 얻기
path.split('\\')
cls_n = path.split('\\')[-2]
cls_n

int(cls_n)


def get_label(path):
    cls_n = path.split('\\')[-2]
    return int(cls_n)


lbl = get_label(path)
lbl

# 데이터 이미지 사이즈 알기

heights = []
widths = []

for path in tqdm(range(len(data_paths))):
    image_pil = Image.open(data_paths[path])
    image = np.array(image_pil)
    h, w = image.shape

    heights.append(h)
    widths.append(w)

plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.hist(heights)
plt.title('Heights')
plt.axvline(np.mean(heights), color='r', linestyle='dashed', linewidth=2)

plt.subplot(122)
plt.hist(widths)
plt.title('Widths')
plt.axvline(np.mean(widhts), color='r', linestyle='dashed', linewidth=2)

plt.show()

# -------------------------------------------------------
# 3. 데이터의 학습에 대한 이해
# -------------------------------------------------------
# cifar data zip 파일 압축 해제
cifar_zip = zipfile.ZipFile(data_dir+'cifar.zip')
cifar_zip.extractall(data_dir+'cifar')
cifar_zip.close()

os.listdir(data_dir+'cifar/cifar/')
data_train_dir = data_dir + 'cifar/cifar/train/'

data_paths = glob(data_train_dir+'*.png')

path = data_paths[0]
path

gfile = tf.io.read_file(path)
image = tf.io.decode_image(gfile, dtype=tf.float32)
image.shape

plt.imshow(image)
plt.show()


def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image


image = read_image(data_paths[1])
image.shape

plt.imshow(image)
plt.show()

# batch
batch_image = []
for path in data_paths[:8]:
    image = read_image(path)
    batch_image.append(image)

plt.imshow(batch_image[1])
plt.show()

batch = tf.convert_to_tensor(batch_image)
batch.shape

# -------------------------------------------------------
# 4. Fit_generator - Image Transformation
# -------------------------------------------------------
data_train_dir = data_dir + 'mnist/mnist_png/training/'
data_paths = glob(data_train_dir+'0/*.png')
data_paths[0]

path = data_paths[0]

# tensorflow에서 glob와 비슷한 기능
data_paths = tf.io.matching_files(data_train_dir+'*/*.png')
data_paths[0]

# Image load
gfile = tf.io.read_file(path)
image = tf.io.decode_image(gfile)
image.shape

# Set Data Generator

image.shape
inputs = image[tf.newaxis, ...]
inputs.shape

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

result = next(iter(datagen.flow(inputs)))

print(np.min(result), np.max(result), np.mean(result))

plt.imshow(result[0, :, :, 0], 'gray')
plt.show()

# Transformation
datagen = ImageDataGenerator(
    width_shift_range=0.3)
outputs = next(iter(datagen.flow(inputs)))

plt.subplot(121)
plt.title('Original Image')
plt.imshow(np.squeeze(inputs), 'gray')

plt.subplot(122)
plt.title('Transformed Image')
plt.imshow(np.squeeze(outputs), 'gray')
plt.show()

datagen = ImageDataGenerator(
    zoom_range=0.5)
outputs = next(iter(datagen.flow(inputs)))

plt.subplot(121)
plt.title('Original Image')
plt.imshow(np.squeeze(inputs), 'gray')

plt.subplot(122)
plt.title('Transformed Image')
plt.imshow(np.squeeze(outputs), 'gray')
plt.show()

# Rescale시 주의사항
# Testset에는 Trainset과 같이 transformation을 해줄 필요없으나 rescale은 해야함
train_datagen = ImageDataGenerator(
    zoom_range=0.7,
    rescale=1./255.)
test_datagen = ImageDataGenerator(
    rescale=1./255.)

# -------------------------------------------------------
# 5. Fit_generator - flow from directory
# -------------------------------------------------------

train_dir = data_dir+'mnist/mnist_png/training/'
test_dir = data_dir+'mnist/mnist_png/testing/'

# Hyperparameter Tunning
num_epochs = 10
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.5
input_shape = (28, 28, 1)
num_classes = 10

# Preprocess
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Build Model
inputs = layers.Input(input_shape)
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(dropout_rate)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(dropout_rate)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(dropout_rate)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')

# Model is the full model w/o custom layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),  # Optimization
              loss='categorical_crossentropy',  # Loss Function
              metrics=['accuracy'])  # Metrics / Accuracy

# Training
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),  # batch와 같은 개념
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# -------------------------------------------------------
# 6. flow_from_dataframe_dataframe 만들기
# -------------------------------------------------------
data_dir
