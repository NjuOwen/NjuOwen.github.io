---
layout: post
title:  "AndrewNg-MachineLearning-Lecture8"
subtitle: “核方法 软间隔 SMO算法 LIBSVM”
date:   2018-02-25
categories: [kernel method,soft margin,SMO algorithm]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**核方法**

之前的讨论都是在样本线性可分的情况下，如果样本线性不可分，则需要将样本映射到高维空间$x\to\phi(x)$，使其线性可分。比如
$$
\phi(x)=\begin{bmatrix} x\\x^2\\x^3\\ \end{bmatrix}
$$
那么此时内积$<x^{(i)},x^{(j)}>$变为$<\phi(x^{(i)}),\phi(x^{(j)})>$,令$k(x^{(i)},x^{(j)})=<\phi(x^{(i)}),\phi(x^{(j)})>=\phi(x^{(i)})^T\phi(x^{(j)})$,则$k$称为核函数。SVM一般采用高斯核替代优化问题中的内积，还有几种其他的核函数：线性核，多项式核，拉普拉斯核，Sigmoid核。于是可以得到
$$f(x)=w^T\phi(x)+b=\Sigma_{i=1}^m\alpha_iy_ik(x,x_i)+b$$
核函数有很多，如果判断核函数的合法性呢？由此引出Mercer定理:对于任意数据D={$x_1,x_2,\cdots,x_m$}，只要对称函数$k(\cdot,\cdot)$对应的核矩阵半正定,它就能作为核函数使用，其中$K_{ij}=k(x_i,x_j)$。而对于一个半正定核矩阵，总能找到一个与之相对应的映射$\phi$

#**软间隔分类器**

有时在样本空间或特征空间即使找到了分界超平面，也很难确定这个结果是否是由过拟合造成的。所谓软间隔是说允许支持向量机在一些样本上出错，那么硬间隔就是说所有样本都必须划分正确。
为此引入松弛变量$\xi_i=l_{0/1}(y_i(w^Tx_i+b)-1)\geq0$,$l_{0/1}$是0/1损失函数
$$
l_{0/1}(z)=\begin{cases} 1,& \text{if z<0}\\0,&\text{o.w.}\end{cases}
$$
但是$l_{0/1}$非凸非连续，数学性质不太好，优化问题不易求解，通常用其他三种替代损失函数:
hinge损失:$l_{hinge}(z)=max(0,1-z)$
指数损失:$l_{exp}(z)=exp(-z)$
对率损失:$l_{log}(z)=log(1+exp(-z))$
将优化问题改写为
$$
min_{w,b,\xi_i}\ \frac{1}{2}||w||^2+C\Sigma_{i=1}^m\xi_i\\
s.t.y_i(w^Tx_i+b)\geq1-\xi_i\\
\xi_i\geq0,i=1,2,\cdots,m.
$$
它的对偶问题是
$$
max_\alpha\ \ \Sigma_{i=1}^m\alpha_i-\frac{1}{2}\Sigma_{i=1}^m\Sigma_{j=1}^my^{(i)}y^{(j)}\alpha_i\alpha_j<x^{(i)}x^{(j)}>\\
s.t.\Sigma_{i=1}^m\alpha_iy_i=0\\
0\leq\alpha\leq C
$$
与硬间隔下的对偶问题对比可以看出，两者的唯一差别在于对于对偶变量的约束不同。

#**SMO算法**

支持向量机的最后一步在于如何求解对偶问题，尽管可以通过二次规划算法求解，但是相比来说用SMO(Sequential Minimal Optimization)算法要高效的多。SMO算法的基本思路是:先固定$\alpha_i$ 之外的所有参数，然后求$\alpha_i$上的极值。由于对偶问题存在约束$\Sigma_{i=1}^m\alpha_iy_i=0$，所以SMO每次选择两个变量$\alpha_i$和$\alpha_j$，并固定其他参数，求关于$\alpha_i$和$\alpha_j$的极值，每次都选一对新的变量直到收敛。
$$
\alpha_iy_i+\alpha_jy_j=c=-\Sigma_{k\neq i,j}\alpha_ky_k\\
\alpha_j=\frac{c-\alpha_iy_i}{y_j}
$$
代入消去$\alpha_j$,则目标函数变为一个关于$\alpha_i$的单变量二次规划问题，仅有约束$\alpha_i\geq 0$,求取极值是非常方便的。
在选取变量的方法上，SMO采用启发式算法，要求每次选取的两个变量所对应样本之间的间隔最大。

#**LIBSVM**

LIBSVM是台湾大学林智仁(Lin Chih-Jen)教授等开发设计的一个简单、易于使用和快速有效的SVM模式识别与回归的软件包,其中有matlab接口。我利用LIBSVM对西瓜数据集3.0$\alpha$用高斯核训练了一个SVM
matlab代码如下
```matlab
clc,clear
%数据读取
x1=[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719];
x2=[0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103];
data=[x1;x2];
label_p=ones(1,8);
label_n=ones(1,9);
label=[label_p,-label_n];
length = size(data,2);
for i =1:length
    if label(i) == 1
        plot(x1(i),x2(i),'bx');
        hold on
    else
         plot(x1(i),x2(i),'rx');
         hold on
    end
end
model = fitcsvm(data',label','KernelFunction','RBF', 'KernelScale','auto');%训练模型
sv=model.SupportVectors;
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);%标记出支持向量
test_point=[0.5,0.05];
predicted_label=predict(model, test_point);%根据模型预测
if predicted_label == 1
    plot(test_point(1,1),test_point(1,2),'bd');
else
    plot(test_point(1,1),test_point(1,2),'rd');
end
```
西瓜数据集3.0$\alpha$
![]()
