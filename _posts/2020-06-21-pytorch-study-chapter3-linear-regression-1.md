---
title: Pytorch 学习笔记 - Linear Regression (1)
date: 2020-06-21 09:58
categories:
- machine learning
- pytorch
- notes
tags:
- machine learning
- pytorch
- 学习笔记
---
读 <<动⼿学深度学习 Pytorch 版>>

## 线性回归
### 模型定义
设房屋的面积为 $x_1$，房龄为 $x_2$，售出价格为 $y$。我们需要建立基于输入 $x_1$ 和 $x_2$ 来计算输出 $y$ 的表达式，也就是模型（model）。顾名思义，线性回归假设输出与各个输入之间是线性关系：


$$
\hat{y} = x_1 w_1 + x_2 w_2 + b
$$

其中 $w_1$ 和 $w_2$ 是权重（weight），$b$ 是偏差（bias），且均为标量。它们是线性回归模型的参数（parameter）。模型输出 $\hat{y}$ 是线性回归对真实价格 $y$ 的预测或估计。

假设我们采集的样本数为 $n$，索引为 $i$ 的样本的特征为 $x_1^{(i)}$ 和 $x_2^{(i)}$，标签为 $y^{(i)}$。对于索引为 $i$ 的房屋，线性回归模型的房屋价格预测表达式为


$$
\hat{y}^{(i)} = x_1^{(i)} w_1 + x_2^{(i)} w_2 + b
$$

### 损失函数
它在评估索引为 $i$ 的样本误差的表达式为


$$\ell^{(i)}(w_1, w_2, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$$


其中常数 $\frac 1 2$ 使对平方项求导后的常数系数为1，这样在形式上稍微简单一些。
我们用训练数据集中所有样本误差的平均来衡量模型预测的质量，即（注：***这里改成 $m$ 样本，原书为 $n$***）


$$
\ell(w_1, w_2, b) =\frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w_1, w_2, b) =\frac{1}{m} \sum_{i=1}^m \frac{1}{2}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right)^2
$$


在模型训练中，我们希望找出一组模型参数，记为 $w_1^*, w_2^*, b^*$，来使训练样本平均损失最小：


$$
w_1^*, w_2^*, b^* = \underset{w_1, w_2, b}{\arg\min} \ell(w_1, w_2, b)
$$


### 矢量计算表达式
如果我们对训练数据集里的3个房屋样本（索引分别为1、2和3）逐一预测价格，将得到


$$
\begin{aligned}
\hat{y}^{(1)} &= x_1^{(1)} w_1 + x_2^{(1)} w_2 + b,\\
\hat{y}^{(2)} &= x_1^{(2)} w_1 + x_2^{(2)} w_2 + b,\\
\hat{y}^{(3)} &= x_1^{(3)} w_1 + x_2^{(3)} w_2 + b.
\end{aligned}
$$


现在，我们将上面3个等式转化成矢量计算。设


$$
\boldsymbol{\hat{y}} =
\begin{bmatrix}
    \hat{y}^{(1)} \\
    \hat{y}^{(2)} \\
    \hat{y}^{(3)}
\end{bmatrix},\quad
\boldsymbol{X} =
\begin{bmatrix}
    x_1^{(1)} & x_2^{(1)} \\
    x_1^{(2)} & x_2^{(2)} \\
    x_1^{(3)} & x_2^{(3)}
\end{bmatrix},\quad
\boldsymbol{w} =
\begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}
$$


对3个房屋样本预测价格的矢量计算表达式为$\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b,$ 其中的加法运算使用了广播机制

延伸到 $m$ 个样本
$$
\begin{aligned}
\hat{y}^{(1)} &= x_1^{(1)} w_1 + x_2^{(1)} w_2 + b,\\
\hat{y}^{(2)} &= x_1^{(2)} w_1 + x_2^{(2)} w_2 + b,\\
\hat{y}^{(3)} &= x_1^{(3)} w_1 + x_2^{(3)} w_2 + b,\\
\cdots,\\
\hat{y}^{(m)} &= x_1^{(m)} w_1 + x_2^{(m)} w_2 + b
\end{aligned}
$$


$$
\boldsymbol{\hat{y}} =
\begin{bmatrix}
    \hat{y}^{(1)} \\
    \hat{y}^{(2)} \\
    \hat{y}^{(3)} \\
    \cdots \\
    \hat{y}^{(m)}
\end{bmatrix},\quad
\boldsymbol{X} =
\begin{bmatrix}
    x_1^{(1)} & x_2^{(1)} \\
    x_1^{(2)} & x_2^{(2)} \\
    x_1^{(3)} & x_2^{(3)} \\
    \cdots \\
    x_1^{(m)} & x_2^{(m)}
\end{bmatrix},\quad
\boldsymbol{w} =
\begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}
$$

__**延伸到 $m$ 个样本，$n$ 个变量 $x$, 注意这里还只是一个输出 $y$**__
$$
\begin{aligned}
\hat{y}^{(1)} &= x_1^{(1)} w_1 + x_2^{(1)} w_2 + \cdots + x_n^{(1)} w_n + b,\\
\hat{y}^{(2)} &= x_1^{(2)} w_1 + x_2^{(2)} w_2 + \cdots + x_n^{(2)} w_n + b,\\
\hat{y}^{(3)} &= x_1^{(3)} w_1 + x_2^{(3)} w_2 + \cdots + x_n^{(3)} w_n + b,\\
\cdots,\\
\hat{y}^{(m)} &= x_1^{(m)} w_1 + x_2^{(m)} w_2 + \cdots + x_n^{(m)} w_n + b
\end{aligned}
$$

$$
\boldsymbol{\hat{y}} =
\begin{bmatrix}
    \hat{y}^{(1)} \\
    \hat{y}^{(2)} \\
    \hat{y}^{(3)} \\
    \cdots \\
    \hat{y}^{(m)}
\end{bmatrix}_{m \times 1},\quad
\boldsymbol{X} =
\begin{bmatrix}
    x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
    x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)}\\
    x_1^{(3)} & x_2^{(3)} & \cdots & x_n^{(3)}\\
    \cdots \\
    x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix}_{ m \times n},\quad
\boldsymbol{w} =
\begin{bmatrix}
    w_1 \\
    w_2 \\
    \cdots \\
    w_n
\end{bmatrix}_{n \times 1}
$$


```python
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from IPython import display
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
```

## 线性回归的从零开始实现
之前参考过一篇从零开始实现线性回归，完全用numpy实现，其中只有一个feature变量，下面是参考原书，有两个features 的线性回归。做了一点小小的改动

### ⽣成数据集


```python
torch.manual_seed(1)
num_samples, num_features, num_outputs = 1000, 2, 1
true_w = torch.randn(num_features, 1)
true_b = torch.rand(1)
true_w, true_b # true w, b
```


    (tensor([[0.6614],
             [0.2669]]),
     tensor([0.0293]))


```python
X = torch.randn(num_samples, num_features) 
y = torch.mm(X, true_w) + true_b + torch.randn(num_samples,1) * 0.003
```


```python
display.set_matplotlib_formats('svg')
fig = plt.figure(figsize=(8,6))
fig = fig.add_subplot(111, projection='3d')

fig.scatter(X[:,0].numpy(), X[:,1].numpy(), y[:,0].numpy())
fig.set_xlabel('X1')
fig.set_ylabel('X2')
fig.set_zlabel('y')
plt.show()
```


![input_data](/assets/images/pytorch_learn/ch3_p4_linear_regression_input.svg)


### 读取数据


```python
def data_iter(batch_size, X, y):
    num_samples = len(X)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size, num_samples)])
        yield X.index_select(0,j), y.index_select(0,j)
        
batch_size = 10
```

### 初始化模型参数


```python
w = torch.randn(num_features, 1, dtype=torch.float32)
b = torch.zeros(1,dtype=torch.float32)
w,b
```




    (tensor([[0.2571],
             [0.2837]]),
     tensor([0.]))




```python
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
```




    tensor([0.], requires_grad=True)



### 定义模型, 损失函数，优化算法


```python
def linreg(X, w, b):
    return torch.mm(X,w) + b

def squared_loss(y_hat, y):
    return (y_hat - y)**2 / 2.

def sgd(params, learning_rate, batch_size):
    for param in params:
        param.data -= learning_rate / batch_size * param.grad
```

### 训练模型


```python
lr = 1e-3
num_epochs = 100
net = linreg
loss = squared_loss
train_loss = []

for epoch in range(num_epochs):
    for bx,by in data_iter(batch_size, X,y):
        l = loss(net(bx,w,b), by).sum()
        l.backward()
        sgd(params=[w,b], learning_rate=lr, batch_size=batch_size)
        
        w.grad.data.zero_()
        b.grad.data.zero_()
        
    train_loss.append(loss(net(X, w, b), y).mean().item())
    
    if (epoch + 1) % 10 == 0: 
        print('Epoch %d, loss %f' % (epoch+1, train_loss[-1]))
        
plt.figure(figsize=(8,6))
plt.title('Cost vs epoch')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.plot(train_loss)
plt.show()
              
w, b
```

    Epoch 10, loss 0.011168
    Epoch 20, loss 0.001537
    Epoch 30, loss 0.000216
    Epoch 40, loss 0.000034
    Epoch 50, loss 0.000009
    Epoch 60, loss 0.000005
    Epoch 70, loss 0.000005
    Epoch 80, loss 0.000004
    Epoch 90, loss 0.000004
    Epoch 100, loss 0.000004



![cost](/assets/images/pytorch_learn/ch3_p4_linear_regression_cost.svg)





    (tensor([[0.6613],
             [0.2670]], requires_grad=True),
     tensor([0.0294], requires_grad=True))



### 预测数据


```python
y_hat = torch.mm(X, w) + b
```


```python
fig = plt.figure(figsize=(8,6))
fig = fig.add_subplot(111, projection='3d')

fig.scatter(X.numpy()[:,0], X.numpy()[:,1], y_hat.detach().numpy()[:,0])
fig.set_xlabel('X1')
fig.set_ylabel('X2')
fig.set_zlabel('predicted y')
plt.show()
```

![input_data](/assets/images/pytorch_learn/ch3_p4_linear_regression_prediction.svg)

### Interactive 3D 图

```python
import plotly.graph_objects as go
import plotly.express as px

# fig = px.scatter_3d(x=X[:,0].numpy(),y=X[:,1].numpy(), z=y[:,0].numpy())
input_data = go.Scatter3d(
    x = X[:,0].numpy(),
    y = X[:,1].numpy(),
    z = y[:,0].numpy(),
    mode = 'markers',
    name = 'Input Data',
    marker = dict(
        color = 'rgb(160, 57, 147)',
        size = 2,
        symbol = 'circle',
        opacity = 0.9
    )
)

predict_data = go.Scatter3d(
    x = X[:,0].numpy(),
    y = X[:,1].numpy(),
    z = y_hat.detach().numpy()[:,0],
    mode = 'markers',
    name = 'Predict Data',
    marker = dict(
        color = 'rgb(50, 168, 139)',
        size = 2,
        symbol = 'circle',
        opacity = 0.9
    )
)

data = [input_data, predict_data]
layout = go.Layout(
    scene = dict(
        xaxis = dict(title='X1'),
        yaxis = dict(title='X2'),
        zaxis = dict(title='y'),
    ),
    margin = dict(
        l=10,
        r=10,
        b=10,
        t=10
    ),
    width = 620,
    height = 440,
)

fig = go.Figure(data = data, layout = layout)
fig.show()
```

<script type="text/javascript">
  function resizeIframe(obj){
     obj.style.height = 0;
     obj.style.height = obj.contentWindow.document.body.scrollHeight  + 'px';
     obj.style.width = obj.contentWindow.document.body.scrollWidth + 'px';
  }
</script>
<iframe src="/assets/images/plotly/figure.html" frameborder="0" scrolling="no"
    onload='resizeIframe(this)'></iframe>
