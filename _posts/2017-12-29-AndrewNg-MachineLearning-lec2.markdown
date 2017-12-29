---
layout: post
title:  "Andrew Ng-Machine Learning-Lecture2 "
subtitle: "线性回归 梯度下降 正规方程"
date:   2017-12-29
categories: [machine leaning,linear regression,gradient descent,normal equation]

---
<script type="text/javascript"
src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
#线性回归

这一讲的主题是线性回归，而线性回归的本质就是利用最小二乘法找到假设函数。这里介绍了梯度下降（微分角度）和正规方程（线代角度）两种方法解决最小二乘法

#正规方程

假设自变量为$X$（$X_0=1$）,因变量为$Y$,假设函数为$Y = \theta^\top X$。根据正规方程即可简洁地求出参数$\theta$。
$$\theta =（X^\top X）X^\top Y$$
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
