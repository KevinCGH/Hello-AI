# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(train_images)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ### 预处理数据
# 必须先对数据进行预处理，然后再训练网络
plt.figure(num=1)
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.savefig("./output/sample-1-figure-1.png", dpi=200, format='png')
plt.show()
plt.close()

train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示训练集中的前25张图像，并在每张图像下显示类别名称
plt.figure(num=2, figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
plt.close()

# 构建模型
# 构建神经网络需要先配置模型的层， 然后再编译模型
# 设置层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 将图像格式从二维数组(28x28像素）转换成一维数组（784像素）
    keras.layers.Dense(128, activation=tf.nn.relu),  # 全连接神经层，具有128哥节点（或神经元）
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 全连接神经层，具有10哥节点的softmax层
])

# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),  # 优化器：根据模型看到的数据及其损失函数更新模型的方式
              loss='sparse_categorical_crossentropy',  # 损失函数： 衡量模型在训练期间的准确率
              metrics=['accuracy'])  # 指标：用于监控训练和测试步骤；这里使用准确率（图像被正确分类的比例）

# 训练模型
# 将训练数据馈送到模型中，模型学习将图像于标签相关联
# 调用model.fit 方法开始训练，使模型于训练数据拟合
model.fit(train_images, train_labels,
          epochs=5,  # 训练周期（训练模型迭代轮次）
          verbose=2  # 日志显示模式： 0为安静模式，1为进度条（默认），2为每轮一行
          )

# 评估准确率
# 比较模型在测试数据集上的表现
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss: {} - Test accuracy: {}'.format(test_loss, test_acc))

# 做出预测
predictions = model.predict(test_images)  # 使用predict() 方法进行预测
print('The first prediction: {}'.format(predictions[0]))  # 查看第一个预测结果（包含10个数字的数据，分别对应10种服饰的"置信度"
label_number = np.argmax(predictions[0])  # 置信度值最大的标签
print("label: {} - class name: {}".format(label_number, class_names[label_number]))
print("Result true or false: {}".format(test_labels[0] == label_number))  # 对比测试标签，查看该预测是否正确


# 可视化：将该预测绘制成图来查看全部10哥通道
def plot_image(m, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[m], true_label[m], img[m]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(n, predictions_array, true_label):
    predictions_array, true_label = predictions_array[n], true_label[n]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 查看第0张图像、预测和预测数组
i = 0
plt.figure(num=3, figsize=(8, 5))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.xticks(range(10), class_names, rotation=45)  # x坐标轴刻度，参数rotation表示label旋转显示角度
plt.show()
plt.close()

# 绘制图像：正确的预测标签为蓝色，错误的为红色，数字表示预测标签的百分比（总计为100)
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(num=5, figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
    plt.xticks(range(10), class_names, rotation=90)
plt.show()
plt.close()
