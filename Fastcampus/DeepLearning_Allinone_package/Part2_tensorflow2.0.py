from tensorflow.keras import layers
import glob
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# -------------------------------------------------------
# 1. Tensor  생성
# -------------------------------------------------------
# Array -> Tensor
arr = np.array([1, 2, 3])
tensor = tf.constant(arr)
tensor.shape
tensor.dtype

# data type 정의
tensor = tf.constant([1, 2, 3], dtype=tf.float32)

# data type 변환
tf.cast(tensor, dtype=tf.uint8)

# tensor에서 numpy 불러오기
tensor.numpy()
np.array(tensor)

# 난수생성
tf.random.normal([3, 3])
tf.random.uniform([3, 3])

# -------------------------------------------------------
# 2. MNIST
# -------------------------------------------------------

# 데이터 불러오기
mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 이미지 확인
image = train_x[0]
plt.imshow(image, 'gray')
plt.show()

# -------------------------------------------------------
# 2.1 채널 확인(batch size, height, width, channel)
# gray scale 이면 1, RGB이면 3으로 만들어줘야함

# 차원 수 늘리기(numpy)
expanded_data = np.expand_dims(train_x, -1)
expanded_data.shape

# tensorflow 패키지로 차원 수 늘리기(tensorflow)
new_train_x = tf.expand_dims(train_x, -1)
new_train_x.shape

# tensorflow 공홈에서 알려준 방법
train_x[..., tf.newaxis].shape

reshaped = train_x.reshape([60000, 28, 28, 1])
reshaped.shape

# 차원 축소하기
np.squeeze(reshaped).shape

# -------------------------------------------------------
# 2.2 Label dataset
train_y.shape
train_y[0]

# -------------------------------------------------------
# 2.3 OneHot Encoding

label = train_y[0]
label_onehot = to_categorical(label, num_classes=10)
label_onehot

# -------------------------------------------------------
# 2.4 Layer Explaination

tf.keras.layers.Conv2D(filters=3, kernel_size=(
    3, 3), strides=(1, 1), padding='SAME', activation='relu')

tf.keras.layers.Conv2D(3, 3, 1, 'SAME')  # 위랑 동일

image = train_x[0][tf.newaxis, ..., tf.newaxis]
image = tf.cast(image, dtype=tf.float32)
image.shape

layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(
    3, 3), strides=(1, 1), padding='SAME')

output = layer(image)

plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(image), 'gray')
plt.subplot(1, 2, 2)
plt.imshow(output[0, :, :, 0], 'gray')
plt.show()

print(np.min(image), np.max(image))
print(np.min(output), np.max(output))

# -------------------------------------------------------
# 2.5 Weight

weight = layer.get_weights()
len(weight)
weight[0].shape, weight[1].shape  # 앞은 layer의 weight, 뒤는 bias

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 500)

plt.subplot(132)
plt.title(weight[0].shape)
plt.imshow(weight[0][:, :, 0, 0], 'gray')

plt.subplot(133)
plt.title(output[0, :, :, 0].shape)
plt.imshow(output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()

# -------------------------------------------------------
# 2.6 Activation function

act_layer = tf.keras.layers.ReLU()
act_output = act_layer(output)
output.shape
np.min(output), np.max(output)
np.min(act_output), np.max(act_output)

# -------------------------------------------------------
# 2.7 Pooling

pool_layer = tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=(2, 2), padding='SAME')
pool_output = pool_layer(act_output)

act_output.shape
pool_output.shape

plt.figfure(figsize=(15, 5))
plt.subplot(121)
plt.hist(pool_output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 100)

plt.subplot(122)
plt.title(pool_output.shape)
plt.imshow(pool_output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()

# -------------------------------------------------------
# 2.8 Fully connected
# flatten
layer = tf.keras.layers.Flatten()
flatten = layer(output)

output.shape
flatten.shape

plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.hist(flatten.numpy().revel())
plt.subplot(212)
plt.imshow(flatten[:, :100], 'jet')
plt.show()

# Dense
layer = tf.keras.layers.Dense(32, activation='relu')
output = layer(flatten)
output.shape

layer2 = tf.keras.layers.Dense(10, activation='relu')
output_example = layer2(output)
output_example.shape

# DropOut
layer = tf.keras.layers.Dropout(0.7)
output = layer(output)
output.shape


# -------------------------------------------------------
# 2.9 Build Model

input_shape = (28, 28, 1)
num_classes = 10

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

inputs = layers.Input(shape=input_shape)
# Feature Extraction
net = layers.Conv2D(32, 3, padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, 3, padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, 3, padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, 3, padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

# Fully Connected
net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')
model.summary()

# -------------------------------------------------------
# 2.10 Optimization & Training(Beginner)
# Loss Function (Categorical vs Binary)
tf.keras.losses.binary_crossentropy
loss = 'binary_crossentropy'
loss = 'categorical_crossentropy'

# Loss Function (sparse_categorical_crossentropy vs categorical_crossentropy)
loss_func = tf.keras.losses.sparse_categorical_crossentropy
tf.keras.losses.categorical_crossentropy

# Metrics
metrics = ['accuracy']
# metrics = [tf.keras.metrics.Accuracy()]

# Compile
optm = tf.keras.optimizers.Adam()
model.compile(optimizer=optm,
              loss=loss_func,
              metrics=metrics)

# Prepare Dataset
train_x.shape, train_y.shape
test_x.shape, test_y.shape

train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]

train_x = train_x/255.
test_x = test_x/255.

# Training
num_epochs = 1
batch_size = 32

model.fit(train_x, train_y,
          batch_size=batch_size,
          shuffle=True,
          epochs=num_epochs)

# -------------------------------------------------------
# 2.11 Check History
hist = model.fit(train_x, train_y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs)

test_image = test_x[0, ..., 0]
test_image.shape

plt.imshow(test_image, 'gray')
plt.title(test_y[0])
plt.show()

pred = model.predict(test_image.reshape(1, 28, 28, 1))
pred.shape

np.argmax(pred)

test_batch = test_x[:32]
test_batch.shape

preds = model.predict(test_batch)
preds.shape

np.argmax(preds, -1).shape
np.argmax(preds, -1)

model.evaluate(test_x, test_y, batch_size=batch_size)


# -------------------------------------------------------
# 3. Modeling - Expert(공홈에서 알려주는 방법)
# -------------------------------------------------------

mnist = tf.keras.datasets.mnist
# Load Data from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Channel 차원 추가
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Data Normalization
x_train, x_test = x_train/255., x_test/255.

# -------------------------------------------------------
# 3.1 tf.data
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(32)

# Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_ds, epochs=100)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


for epoch in range(2):
    print('Start Traning')
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy:{}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
