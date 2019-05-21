# Learn and use ML

## Program

### 1.执行fashion_mnist.load_data()失败
该问题是因为网络原因无法下载训练和测试数据。
手动把文件下载到本地`~/.keras/datasets/fashion-mnist`目录下
```bash
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
```