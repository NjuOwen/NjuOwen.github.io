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
[2ex]0 &\text{o.w.}
\end{cases}
$$
离散化举例:将住房面积这个特征划分为离散的数值
$$
\begin{array}{c|c}
Living\ area & <500 & 500\sim1000 & 1000\sim1500& >1500\\
\hline
 x_i & 1 & 2 & 3 & 4
\end{array}
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
