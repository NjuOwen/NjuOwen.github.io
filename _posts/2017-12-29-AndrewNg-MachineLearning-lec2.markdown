---
layout: post
title:  "AndrewNg-MachineLearning-Lecture2 "
subtitle: "线性回归 梯度下降 正规方程"
date:   2017-12-29
categories: [machine leaning,linear regression,gradient descent,normal equation]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
#线性回归

这一讲的主题是线性回归，而线性回归的本质就是利用最小二乘法找到假设函数。这里介绍了梯度下降（微分角度）和正规方程（线代角度）两种方法解决最小二乘法

#正规方程

假设自变量为$X$（$X_0=1$）,因变量为$Y$,假设函数为$Y = \theta^\top X$。根据正规方程$\theta =（X^\top X）X^\top Y$即可简洁地求出参数$\theta$。
matlab代码如下：
```matlab
%房屋价格Y
Y = [2300,4500,2300,5200,2900,8000,5000,5000,3200,2800,3200,4000,2600,2400,3600]';
%房屋面积s
s = [52,131,49,158,66,143,140,76,84,46,51,111,66,90,52]';
%卧室数量b
b = [1,3,1,4,2,3,3,2,2,2,2,3,2,2,2]';
x0 = ones(15,1);
X = [x0,s,b];
theta = inv(X'*X)*X'*Y;
%画散点图
%plot3(s,b,Y,'bo');
%画出三维图
x = 46:0.1:158;%房间面积
y = 1:0.1:4;%卧室数量
[x,y] = meshgrid(x,y);
z = theta(1)+theta(2)*x+theta(3)*y;
mesh(x,y,z);
%测试 测试点 房屋价格：3300 房屋面积：57 卧室数量：2 房屋年龄：36
%y_hat =  theta(1)+theta(2)*57+theta(3)*2
```
预测模型如下 自变量：房屋面积，卧室数 应变量：房租价格  

 ![](https://github.com/NjuOwen/NjuOwen.github.io/blob/master/%E6%AD%A3%E8%A7%84%E6%96%B9%E7%A8%8B%E6%95%88%E6%9E%9C%E5%9B%BE.JPG)


#梯度下降
设$\theta$的初始值为$\mathbf0$,样本个数为m，因为导数方向为下降最快方向，所以
通过$\theta_i := \theta_i -\alpha\Sigma_{j=1}^m(h_\theta(x^{(j)})-y^{(j)})\cdot x_i^{(j)}$进行迭代，直到前后两次结果产生的$\theta$的欧氏距离小于某一阈值，结束迭代。但是需要指出的是，梯度下降方法相对于正规方程，结果必然存在误差。阈值和学习速度$\alpha$也都需要调试。
假设只有一个自变量：房屋面积，$\theta$的有两个参数，此时方面形象地展示梯度下降法。
matlab代码如下:
```matlab
%房屋价格Y
Y = [2300,4500,2300,5200,2900,8000,5000,5000,3200,2800,3200,4000,2600,2400,3600]';
%房屋面积s
s = [52,131,49,158,66,143,140,76,84,46,51,111,66,90,52]';
%卧室数量b
b = [1,3,1,4,2,3,3,2,2,2,2,3,2,2,2]';
%房屋年龄n
%n = [4,8,14,11,11,7,1,9,7,27,7,10,7,10,27]';
x0 = ones(15,1);
X = [x0,s,b];
theta = inv(X'*X)*X'*Y;
%画散点图
%plot3(s,b,Y,'bo');
%画出三维图
x = 46:0.1:158;%房间面积
y = 1:0.1:4;%卧室数量
[x,y] = meshgrid(x,y);
z = theta(1)+theta(2)*x+theta(3)*y;
mesh(x,y,z);
```
$\theta$从$\mathbf0$，不断进行梯度下降，直到滑落到代价函数$J(\theta)$的最低点
![](C:/Users/文昊/Documents/GitHub/NjuOwen.github.io/img/2017-12-29-AndrewNg-MachineLearning-lec2/单自变量梯度下降.JPG)

当自变量为房屋价格，卧室数量时
matlab代码如下：
```matlab
%假设房屋价格由房屋面积，卧室数量构成
%theta有3个参数
%==================数据===============
%房屋价格Y
Y = [2300,4500,2300,5200,2900,8000,5000,5000,3200,2800,3200,4000,2600,2400,3600]';
%房屋面积s
s = [52,131,49,158,66,143,140,76,84,46,51,111,66,90,52]';
%卧室数量b
b = [1,3,1,4,2,3,3,2,2,2,2,3,2,2,2]';
%房屋年龄n
%n = [4,8,14,11,11,7,1,9,7,27,7,10,7,10,27]';
%====================================
x0 = ones(15,1);
X = [x0,s,b];
a = 0.000005;
theta = zeros(3,1);%初始化theta
new_theta = zeros(3,1);
d = 100;
%梯度下降法，直到收敛
while (d>0.00001)
            new_theta(1) = theta(1) - a*(sum((X*theta-Y).*X(:,1)));
            new_theta(2) = theta(2) - a*(sum((X*theta-Y).*X(:,2)));
            new_theta(3) = theta(3) - a*(sum((X*theta-Y).*X(:,3)));
            dst = [theta';new_theta'];
            d = pdist(dst,'euclidean');
            theta = new_theta;
end
%画散点图
%plot3(s,b,Y,'bo');
%画出三维图
x = 46:0.1:158;%房屋面积
y = 1:0.1:4;%卧室数量
[x,y] = meshgrid(x,y);
z = theta(1)+theta(2)*x+theta(3)*y;
mesh(x,y,z);
%测试 测试点 房屋价格：3300 房屋面积：57 卧室数量：2 房屋年龄：36
%y_hat =  theta(1)+theta(2)*57+theta(3)*2
```
样本散点图：
![](C:/Users/文昊/Documents/GitHub/NjuOwen.github.io/img/2017-12-29-AndrewNg-MachineLearning-lec2/多变量梯度下降_散点图.JPG)
预测模型：
![](C:/Users/文昊/Documents/GitHub/NjuOwen.github.io/img/2017-12-29-AndrewNg-MachineLearning-lec2/多自变量梯度下降.JPG)
