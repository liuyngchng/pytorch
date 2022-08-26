# 1. 机器学习基本思想

有一组样本数据 ,假定为 $n$ 组， 如下所示：

```python
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 , 10.0]
y = [2.0, 4.1, 6.3, 7.9, 9.8, 11.5, 14.3, 16.6, 18.1, 21]
```

即：
$$
x_1 = 1.0,  x_2=2.0,...\\
y_1 = 2.0, y_2 = 4.1,....
$$
 需要推断一个函数，能够满足这一组样本值的特征，这个推断的过程叫做**机器学习**。

从 $[x, y]$ 的散点图来看，感觉是个一元一次方程（即采用了**线性模型**），假定为
$$
y=wx+b \tag 1
$$
只要能计算出w, b，那么就得到了这个函数，同时从数据分布来看，求得的公式（1）为一条直线，那么对于某个给定的值$x_i$，计算出的值 $y^`_i$ 跟样本集中对应的 $y_i$ 是有误差的 。通常通过均方差来评价这种误差，即
$$
loss = \frac{1}{n}\sum^{n}_{i=1}(y^,_i-y_i)^2\tag 2
$$
公式（2）我们称之为损失函数。需要求解的就是组合
$$
[w, b]\tag 3
$$
使得公式（2）中的 $loss$ 达到其最小值。为了便于说明问题，将公式（1）代入公式（2），即
$$
\begin{align}
loss &= \frac{1}{n}\sum^{n}_{i=1}(y^,_i-y_i)^2\\
&=\frac{1}{n}\sum^{n}_{i=1}(wx_i+b-y_i)^2
\tag 5
\end{align}
$$
公式（5）中，$n,x_i, y_i$ 都是由样本集决定的，即为已知参数。$w, b$ 为未知参数。为了简化问题方便推导，只使用样本集中的 2个样本值，即 $n = 2$， 则： 
$$
\begin{align}
loss &= \frac{1}{2}[(w\times1.0 + b -2.0)^2+(w\times2.0 + b -4.1)^2]\\
&=\frac{1}{2}[(w+b-2)^2+(2w+b-4.1)^2]\\
&=(推导省略...）\\
&=5w^2+2b^2+6wb-20.04w-12.2b+20.81
\end{align}\tag6
$$
问题转化为求公式（6）这个函数的最小值了，将公式（6）中的$loss$ 换成 $z$,  $w,b$分别换成 $x, y$,（只是个人习惯而已，不换也可以），即：
$$
z=5x^2+2y^2+6xy-20.0x-12.2y+20.81\tag7
$$
求公式（7）的最小值。

# 2. 梯度下降法

​     求解公式（7）的函数值 $z$ 最小时，对应的 $x, y$ 的值，即公式（1）中的 $w, b$ 的值，就得到了符合样本数据特征的最佳函数。

​     求解的过程，引入了**梯度下降法**。




