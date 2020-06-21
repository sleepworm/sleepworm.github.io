---
title: Pytorch 学习笔记 - Tensor
date: 2020-06-20 20:34
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
```


```python
x = torch.ones(2,2, requires_grad=True)
print(x)
print(x.grad_fn)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    None



```python
y = x + 2
print(y)
print(y.grad_fn)
print(x.is_leaf, y.is_leaf)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    <AddBackward0 object at 0x7fa9987b8810>
    True False



```python
z = y* y * 3
out = z.mean()
print(z, out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)



```python
out.backward()
print(x.grad)
print(y.grad)
```

    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])
    None





数学上，如果有一个函数值和自变量都为向量的函数 $\vec{y}=f(\vec{x})$, 那么 $\vec{y}$ 关于 $\vec{x}$ 的梯度就是一个雅可比矩阵(Jacobian matrix）: 


$$
J=\left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)
$$


而``torch.autograd``这个包就是用来计算一些雅克比矩阵的乘积的。例如，如果 $v$ 是一个标量函数的 $l=g\left(\vec{y}\right)$ 的梯度：


$$
v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)
$$


那么根据链式法则我们有 $l$ 关于 $\vec{x}$ 的雅克比矩阵就为:


$$
v J=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right) \left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
    \vdots & \ddots & \vdots\\
    \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
    \end{array}\right)=\left(\begin{array}{ccc}\frac{\partial l}{\partial x_{1}} & \cdots & \frac{\partial l}{\partial x_{n}}\end{array}\right)
$$


注意:grad在反向传播过程中是累加的(accumulated)，这意味着每⼀一次运⾏行行反向传播，梯度都会累 加之前的梯度，所以⼀一般在反向传播之前需把梯度清零。

再来反向传播⼀一次，注意grad是累加的


```python
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_() # clear to zero
out3.backward()
print(x.grad)
```

    tensor([[5.5000, 5.5000],
            [5.5000, 5.5000]])
    tensor([[1., 1.],
            [1., 1.]])


假设 x 经过一番计算得到 y，那么 y.backward(w) 求的不是 y 对 x 的导数，而是 l = torch.sum(y*w) 对 x 的导数。w 可以视为 y 的各分量的权重，也可以视为遥远的损失函数 l 对 y 的偏导数（这正是函数说明文档的含义）。特别地，若 y 为标量，w 取默认值 1.0，才是按照我们通常理解的那样，求 y 对 x 的导数。
[reference](https://zhuanlan.zhihu.com/p/29923090)


```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y=2*x
z = y.view(2, 2)
print(z)
```

    tensor([[2., 4.],
            [6., 8.]], grad_fn=<ViewBackward>)



```python
# 现在 y 不不是⼀一个标量量，所以在调⽤用 backward 时需要传⼊入⼀一个和 y 同
# 形的权向量量进⾏加权求和得到 ⼀一个标量量。
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)
```

    tensor([2.0000, 0.2000, 0.0200, 0.0020])


```python
# 再来看看中断梯度追踪的例例⼦
x = torch.tensor(1.0, requires_grad=True)
y1=x**2
with torch.no_grad():
    y2 = x ** 3
y3=y1+y2

print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True
```

    True
    tensor(1., grad_fn=<PowBackward0>) True
    tensor(1.) False
    tensor(2., grad_fn=<AddBackward0>) True
