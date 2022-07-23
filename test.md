# 求导

## 平方求导

（1）已知：
$$
y=f(x) \\
f(x)=x^2+1
$$
（2）求解：
$$
\frac{dy}{dx} =f'(x)= \lim_{\Delta x \to 0}\frac{\Delta y}{\Delta x}
$$
（3）解答
$$
\begin{align}
\frac{\Delta y}{\Delta x} &= \frac{f(x+\Delta x)-f(x)}{\Delta x}\\
&=\frac{[(x+\Delta x)^2 + 1]-[x^2+1]}{\Delta x} \\
&= \frac{(x^2 + 2\Delta x*x + \Delta x^2 +1)-(x^2+1)}{\Delta x}\\
&=\frac{2\Delta x \cdot x + \Delta x^2}{\Delta x} \\ 
&= 2x+\Delta x
\end{align}
$$


$$
\begin{align}
\frac{dy}{dx} &= \lim_{\Delta x \to 0}\frac{\Delta y}{\Delta x}\\ 
&= \lim_{\Delta x \to 0}(2x+\Delta x) \\
&= 2x
\end{align}
$$


## 开方求导
（1）已知：

$$
y=f(x)\newline f(x)=\sqrt{x}
$$
（2）求解：
$$
f'(x) = ?
$$


（3）解答：
$$
\begin {align}
\frac{\Delta y}{\Delta x} &= \frac{f(x+\Delta x)-f(x)}{\Delta x}\\\\
&=\frac{\sqrt{x+\Delta x}-\sqrt{x}}{\Delta x}
\end {align}
$$
两边都乘以 $y$ , 换算一下
$$
\begin{align}
\frac{\Delta y}{\Delta x}\cdot y &=\frac{\sqrt{x+\Delta x }- \sqrt{x}}{\Delta x}\cdot \sqrt{x}\\
&=\frac{\sqrt{x+\Delta x}\cdot\sqrt{x}- x}{\Delta x}\\
&=\frac{\sqrt{x^2+\Delta x\cdot x}- x}{\Delta x}
\end{align}
$$

这儿干不下去了，<font color='red'>求救~~~</font>



