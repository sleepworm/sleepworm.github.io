---
title: Pytorch 学习笔记 - Linear Regression (2)
date: 2020-06-21 16:52
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

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data

from IPython import display

%matplotlib inline
```

### ⽣成数据集


```python
torch.manual_seed(1)
num_samples, num_features, num_outputs = 1000, 2, 1
true_w = torch.rand(num_features, 1)
true_b = torch.rand(1)
true_w, true_b # true w, b
```




    (tensor([[0.7576],
             [0.2793]]),
     tensor([0.4031]))




```python
X = torch.randn(num_samples, num_features) 
y = torch.mm(X, true_w) + true_b + torch.randn(num_samples,1) * 0.003
```

### 读取数据


```python
dataset = Data.TensorDataset(X, y)
data_iter = Data.DataLoader(dataset, batch_size=10, shuffle=True)
```

### 定义模型


```python
class LinearNet(nn.Module):
    def __init__(self, num_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_features, 1) # 1 output
        
    def forward(self, x):
        x = self.linear(x)
        return x
net = LinearNet(num_features)
print(net, '\n', net.linear, '\n', net.linear.weight, '\n', net.linear.bias)
```

    LinearNet(
      (linear): Linear(in_features=2, out_features=1, bias=True)
    ) 
     Linear(in_features=2, out_features=1, bias=True) 
     Parameter containing:
    tensor([[-0.0979, -0.0225]], requires_grad=True) 
     Parameter containing:
    tensor([-0.1533], requires_grad=True)



```python
# # 写法一
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
#     # 此处还可以传入其他层
#     )

# # 写法二
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......

# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#           ('linear', nn.Linear(num_inputs, 1))
#           # ......
#         ]))

# print(net)
# print(net[0])
```


```python
for param in net.parameters():
    print(param)
```

    Parameter containing:
    tensor([[-0.0979, -0.0225]], requires_grad=True)
    Parameter containing:
    tensor([-0.1533], requires_grad=True)


注意: torch.nn 仅⽀持输⼊一个batch的样本不⽀持单个样本输入，如果只有单个样本，可使用 input.unsqueeze(0) 来添加一维。

###  初始化模型参数


```python
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  
# 也可以直接修改bias的data: net[0].bias.data.fill_(0)
print(net.linear.weight, net.linear.bias)
```

    Parameter containing:
    tensor([[ 0.0042, -0.0070]], requires_grad=True) Parameter containing:
    tensor([0.], requires_grad=True)


### 定义损失函数



```python
loss = nn.MSELoss()
```

### 定义优化算法


```python
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=1e-3)
print(optimizer)

```

    SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.001
        momentum: 0
        nesterov: False
        weight_decay: 0
    )


我们还可以为不同子网络设置不同的学习率，这在finetune时经常用到。例：



```python
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)

```

### 训练模型


```python
num_epochs = 100
train_loss = []
for epoch in range(num_epochs):
    for bx, by in data_iter:
        output = net(bx)
        l = loss(output, by)
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    with torch.no_grad():
        train_loss.append(loss(net(X), y).item())
    if (epoch + 1) % 10 == 0:
        print('epoch %d, loss: %f' % (epoch, train_loss[-1]))
```

    epoch 9, loss: 0.012866
    epoch 19, loss: 0.000209
    epoch 29, loss: 0.000012
    epoch 39, loss: 0.000009
    epoch 49, loss: 0.000009
    epoch 59, loss: 0.000009
    epoch 69, loss: 0.000009
    epoch 79, loss: 0.000009
    epoch 89, loss: 0.000009
    epoch 99, loss: 0.000009



```python
plt.figure(figsize=(8,6))
plt.plot(train_loss)
plt.show()
```


![cost](/assets/images/pytorch_learn/ch3_p4_linear_regression_pytorch_21_0.svg)



```python
print(net.linear.weight, net.linear.bias)
```

    Parameter containing:
    tensor([[0.7577, 0.2792]], requires_grad=True) Parameter containing:
    tensor([0.4031], requires_grad=True)

### 3D数据图

```python
import plotly.graph_objects as go

input_data = go.Scatter3d(
    x = X[:,0].numpy(),
    y = X[:,1].numpy(),
    z = y[:,0].numpy(),
    mode = 'markers',
    name = 'Input Data',
    marker = dict(
        color = 'rgb(160, 57, 147)',
        size = 4,
        symbol = 'circle',
        opacity = 0.9
    )
)

with torch.no_grad():
    y_hat = net(X)
    
predict_data = go.Scatter3d(
    x = X[:,0].numpy(),
    y = X[:,1].numpy(),
    z = y_hat.detach().numpy()[:,0],
    mode = 'markers',
    name = 'Predict Data',
    marker = dict(
        color = 'rgb(50, 168, 139)',
        size = 2,
        symbol = 'square',
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
<iframe src="/assets/images/plotly/ch3_p4_linear_regression_pytorch.html" frameborder="0" scrolling="no"
    onload='resizeIframe(this)'></iframe>
