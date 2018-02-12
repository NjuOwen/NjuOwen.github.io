---
layout: post
title:  "AndrewNg-MachineLearning-Lecture6 "
subtitle: “朴素贝叶斯方法 神经网络 最大间隔分类器”
date:   2018-02-12
categories: [Naive Bayes,Neural Network,Maximum Margin Classifier]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**朴素贝叶斯方法**

对于朴素贝叶斯方法还有两类变化形式：
（1）$x_i$可以取多个值，$x_i\in \lbrace 1,2,\cdots,k\rbrace$，当特征$x_i$是连续值时，则将其离散化，一般取10个区间。这时$p(x|y)=\Pi_{i=1}^{n}p(x_i|y)$是一个多项式分布。
多项式分布概率公式
$$
P(X_1=x_1,\cdots,X_k=x_k)=
\begin{cases}\frac{n!}{x_1!\cdots x_k!}{p_1^{x_1}\cdots p_k^{x_k}}\ &\text{when$\ \Sigma_{i=1}^{k}x_i=n$ }\\
0 &\text{o.w.}
\end{cases}\\
$$
离散化举例:将住房面积这个特征划分为离散的数值
$$
\begin{array}{c|c}
Living\ area & <500 & 500\sim1000 & 1000\sim1500& >1500\\
\hline
 x_i & 1 & 2 & 3 & 4
\end{array}\\
$$
（2）多项式事件模型
用一个长度等于邮件字数的向量表示一封邮件，向量中的数对应该字在字典中的编号。模型如下：
$$
x=(x_1^{(i)},x_2^{(i)},\cdots x_{n_i}^{(i)})\\
x_j\in\lbrace 1,2,\cdots,50000\rbrace\\
p(x,y)=(\Pi_{i=1}^{n}p(x_i|y))\cdotp(y)\\
\phi_{k|y=1}=p(x_j=k|y=1)\\
\phi_{k|y=0}=p(x_j=k|y=0)\\
\phi_y=p(y=1)\\
$$
对参数作出极大似然估计:
$$
\phi_{k|y=1}=\frac{\Sigma_{i=1}^{m}I\lbrace y^{(i)} = 1 \rbrace\Sigma_{j=1}^{n_i}I\lbrace x_j^{(i)}=k \rbrace}{\Sigma_{i=1}^{m}I\lbrace y^{(i)=1} \rbrace\cdot n_i}\\
\phi_{k|y=0}=\frac{\Sigma_{i=1}^{m}I\lbrace y^{(i)} = 0 \rbrace\Sigma_{j=1}^{n_i}I\lbrace x_j^{(i)}=k \rbrace}{\Sigma_{i=1}^{m}I\lbrace y^{(i)=0} \rbrace\cdot n_i}\\
\phi_y=\frac{\Sigma_{i=1}^{m}I\lbrace y^{(i)=1} \rbrace}{m}
$$

#**神经网络**

![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2018-02-12-AndrewNg-MachineLearning-lec6/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.png)
之前讨论的分类器都是线性分类器，即分界面都是线性的。如果目标分界面是非线性的，怎么构成一个非线性分类器呢？神经网络是其中的一种方法。
$x_1,x_2,x_3\cdots$等特征作为输入，进入隐藏层$a_1,a_2,a_3$，其中
$$
a_1=g(x^T \theta^{(1)})\\
a_2=g(x^T \theta^{(2)})\\
a_3=g(x^T \theta^{(3)})\\
g(z)=\frac{1}{1+e^{-z}}\\
J(\theta)=\frac{1}{2}\Sigma_{i=1}^{m}(y^{i}-h_\theta(x^{(i)}))^2
$$
最终在输出层得到输出结果$h_{\theta}=g(\overrightarrow{a}\cdot\theta)$

#**最大间隔分类器**

虽然神经网络可以用来作为非线性分类器，但是实际操作时非常繁琐，SVM支持向量机往往时更好的选择。在介绍SVM前，先讨论两个概念：函数间隔和几何间隔。
定义模型
$$
y\in \lbrace -1,+1 \rbrace\\
g(z)=
\begin{cases}
1 & \text{if $z\geq 0$}\\
-1 & \text{o.w.}
\end{cases}\\
w\in R^N\\
h_{w,b}(x)=g(w^{T}x+b)
$$
那么样本点$(x^{(i)},y^{(i)})$关于超平面$(w,b)$的**函数间隔**如下：
$$
\hat{\gamma}^{(i)}=y^{(i)}(w^Tx^{(i)}+b)\\
\hat{\gamma}=min\ \hat{\gamma}^{(i)}\\
$$
如果当$y^{(i)}=1$时，$w^Tx^{(i)}+b>>0$,当$y^{(i)}=0$时，$w^Tx^{(i)}+b<<0$,则认为这是一个好的分类器。如果$\hat{\gamma}^{(i)}>0$则表明分类器能够进行正确的分类。
样本点$(x^{(i)},y^{(i)})$关于超平面$(w,b)$的**几何间隔**如下：
作样本点到超平面得到一个交点，距离为$\gamma^{(i)}$,超平面的法向量为$\frac{w}{||w||}$,那么交点为$x^{(i)}-\gamma^{(i)}\frac{w}{||w||}$。将交点带入超平面方程。得到$w^T(x^{(i)}-\gamma^{(i)}\frac{w}{||w||})+b=0$解得几何间隔$\gamma^{(i)}=(\frac{w}{||w||})^Tx+\frac{b}{||w||}，观察得\gamma^{(i)}=\frac{\hat{\gamma}^{(i)}}{||w||}$。一般地，$\gamma =min\ y^{(i)}\cdot[(\frac{w}{||w||})^Tx+\frac{b}{||w||}]$
通过几何间隔的定义(一般规定$||w||=1$)，可以通过一个优化问题推导出一个最大间隔分类器，但是它依然是一个线性分类器。
$$
\underset{\gamma}m\underset{w}a\underset{b}x \ \gamma\\
s.t. \ y^{(i)}(w^Tx^{(i)}+b)\geq\gamma
$$
