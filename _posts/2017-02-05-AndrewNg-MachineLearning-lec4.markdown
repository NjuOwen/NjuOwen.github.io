---
layout: post
title:  "AndrewNg-MachineLearning-Lecture4 "
subtitle: “牛顿法 指数分布族 广义线性回归”
date:   2017-02-05
categories: [machine leaning,newtons method,exponential family]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#牛顿法
之前在寻找使代价函数最小的参数时，都在使用梯度下降法，而本次将介绍牛顿法，梯度下降法是一阶收敛的，牛顿法是二阶收敛的，因此牛顿法比梯度下降法快的多，从实际操作中也是可以看的出来的。先来看最简单的牛顿法，对于$f(\theta)$要找到$\theta$，使得$f(\theta)=0$。首先将 $\theta$ 初始化为 $\theta^{(0)}$,令$\Delta=\frac{f(\theta^{(0)})}{f'(\theta^{(0)})}$,对$
\theta$进行一次更新:$\theta^{(1)} = \theta^{(0)} - \Delta$。一般地，更新算式为:$\theta^{(t+1)} = \theta^{(t)} - \frac{f(\theta^{(t)})}{f'(\theta^{(t)})}$。那么现在， 对于似然函数$L(\theta)$,要找到$\theta$,使得$L'(\theta)=0$，则更新算法为$\theta^{(t+1)} = \theta^{(t)} - \frac{L'(\theta^{(t)})}{L''(\theta^{(t)})}$。
更一般地，推广到多元：$\theta^{(t+1)} = \theta^{(t)} - H^{-1}\nabla L$，其中$H$为Hessian矩阵,也可写为$\nabla^2 L$。$\nabla L$为梯度矩阵,
$$\nabla L = \begin{bmatrix} \frac{dL}{dx_1}\\ \frac{dL}{dx_2}\\ \vdots\\ \frac{dL}{dx_n}\end{bmatrix}     \nabla^2 L = \begin{bmatrix} \frac{d^2L}{dx_1^2}&\frac{d^2L}{dx_1dx_2} &\cdots &\frac{d^2L}{dx_1dx_n}\\ \frac{d^2L}{dx_2dx_1}&\frac{d^2L}{dx_2^2} &\cdots &\frac{d^2L}{dx_2dx_n}\\ \vdots & \vdots & \ddots &\vdots\\ \frac{d^2L}{dx_ndx_1}&\frac{d^2L}{dx_ndx_2} &\cdots &\frac{d^2L}{dx_n^2}\\ \end{bmatrix}$$
将之前用梯度下降法解决的问题用牛顿法解决，最优解与之前一样，但是求解速度快的多。

matlab代码如下：
```matlab
clc,clear
%假设房屋价格由房屋面积，卧室数量构成
%theta有3个参数
%==================数据===============
%房屋价格Y
Y = [2300,4500,2300,5200,2900,8000,5000,5000,3200,2800,3200,4000,2600,2400,3600]';
%房屋面积s
s = [52,131,49,158,66,143,140,76,84,46,51,111,66,90,52]';
%卧室数量b
b = [1,3,1,4,2,3,3,2,2,2,2,3,2,2,2]';
%====================================
x0 = ones(15,1);
X = [x0,s,b];
a = 0.000005;
syms t1 t2 t3;
t = [t1;t2;t3];
J = 0.5*sum((X*t-Y).^2);%代价函数，目标：J‘(theta)=0，求出theta使J最小
theta = zeros(3,1);%初始化theta
new_theta = zeros(3,1);
d = 100;
%牛顿法，直到收敛
while (d>0.00001)
            HM = double(subs(hessian(J,[t1,t2,t3]),{t1,t2,t3},{theta(1),theta(2),theta(3)}));%计算hessian矩阵
            JM = double(subs(jacobian(J,[t1,t2,t3]),{t1,t2,t3},{theta(1),theta(2),theta(3)}));%计算jacobian矩阵
            new_theta = theta - HM\JM';
            dst = [theta';new_theta'];
            d = pdist(dst,'euclidean');
            theta = new_theta;
end
theta%输出theta的值
```
