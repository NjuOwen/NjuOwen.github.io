---
layout: post
title:  "AndrewNg-MachineLearning-Lecture3 "
subtitle: “局部加权回归 逻辑回归 多元分类”
date:   2017-12-30
categories: [machine leaning,locally weighted regression,logistic regression,classification]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#局部加权回归

接着上一讲的线性回归继续，现实生活中很多情况并不可以作简单的一次线性回归，有时必须用高次线性回归，但在选取次数上十分麻烦，次数选取不好就会出现欠拟合或是过拟合的情况。为了规避这麻烦，可以采用局部加权回归。通过权值函数$W^{(i)}=e^{-\frac{(x^{(i)}-x)^2}{2\tau^2}}$(钟形函数)对代价函数$J(\theta)$进行修正，削弱远离预测点的样本点对回归结果的影响。
LWR方法的目标是对每一预测点确定参数$\theta$，使得代价函数$J(\theta)=\Sigma_iW^{(i)}(y^{(i)}-\theta^TX^{(i)})^2$最小。由正规方程可以解出$\theta = (X^TWX)^{-1}X^TWY$($W$为$m*m$对角阵)。
matlab代码如下：
```matlab
%局部加权回归LWR
%房屋价格Y
Y = [1.17,1.88,0.34,2.10,1.64,2.36,2.12,-0.79,2.03,1.97,1.47,2.46,1.98,1.12,-1.37,1.03,1.38,1.22,1.40,-1.52,2.39,1.37,-0.99,2.04,-0.71,1.41,0.14,0.40,0.39,1.39,2.15,-1.6,-0.56,1.44,2.00,1.56,0.92,-0.30,1.14,1.59]';
x = [1.24,2.33,0.13,2.36,6.73,3.70,11.85,-1.87,4.50,3.27,1.75,3.37,11.47,9.05,-2.81,9.31,8.42,0.86,7.55,-3.98,4.49,8.30,-2.60,4.48,-1.50,9.62,-0.62,-0.38,-0.1,7.49,3.44,-4.07,-1.74,7.31,11.55,6.97,8.62,-1.49,8.13,9.83]';
%Y = [2300,4500,2300,5200,2900,8000,5000,5000,3200,2800,3200,4000,2600,2400,3600]';
%房屋面积s
%s = [52,131,49,158,66,143,140,76,84,46,51,111,66,90,52]';
m=size(x,1);%样本数
x0 = ones(m,1);
X = [x0,x];
n=size(X,2);%特征维数
t=[0.3,0.8,2,10,50];%带宽参数
color = ['r','g','m','y','k'];
plot(x,Y,'bo');%画出散点图
hold on;
temp_X = min(X(:,2)) : 0.1 : max(X(:,2));
temp_Y = zeros(1,size(temp_X,2));
for c = 1:size(color,2)
for k = 1:size(temp_X,2)
    W=zeros(m,m);
    for i = 1:m %重新遍历样本
        W(i,i)=exp(-((temp_X(k)-X(i,2))^2)/(2*t(c)^2));
    end
    theta = (X'*W*X)\(X'*W*Y);
    temp_Y(k) = theta(1)+theta(2)*temp_X(k);
end
plot(temp_X,temp_Y,color(c));
end
legend('training data','t = 0.3','t = 0.8','t = 2',' t = 10','t = 50');

```
预测模型：
![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2017-12-30-AndrewNg-MachineLearning-lec3/fig1.JPG)
在LWR方法中，$\tau$的选取对结果至关重要。当$\tau=50$时，出现欠拟合。当$\tau=0.3$时，出现过拟合。而当$\tau=0.8$或$\tau=2$时，结果非常好的。可以这样理解，从权值函数$W$出发，当$\tau$很小时，距离预测点较远的样本点的代价函数权值很小，易造成过拟合。而当$\tau$很大时，距离预测点较远的样本点的代价函数权值依然很大，此时LWR与简单的线性回归几乎无差别，易造欠拟合。

#逻辑回归

逻辑回归与线性回归的思路非常类似，只是$Y$值只能取0或者1，我们可以认为0，1是标签值，那么逻辑回归其实是在做分类。此时假设函数为$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$,其中$g(z)=\frac{1}{1+e^{-z}}$。那么$P(y=1|x;\theta)=h_\theta(x)$(给定$x$在$\theta$下，$y=1$的概率)。$P(y=0|x;\theta)=1-h_\theta(x)$(给定$x$在$\theta$下，$y=0$的概率)。可合并写为$P(y|x;\theta)=h_\theta(x)^y\cdot(1-h_\theta(x))^{1-y}$。此时参数似然性$L(\theta)=P(y|x;\theta)=\prod_iP(y^{(i)}|x^{(i)};\theta)$，$L(\theta)$达到最大时找到极大似然参数$\theta$。利用梯度上升法求解，$\theta := \theta+\alpha\nabla_\theta l(\theta)$,其中$\frac{\mathrm{d}}{\mathrm{d}\theta_j} l(\theta) = \Sigma_{i=1}^m (y^{(i)}-h_\theta(x^{(i)})) \cdot x_j^{(i)}$
matlab代码如下
```matlab
clc,clear
x_0 = [2,0.5,1,2,3,4,1.5,1.2]';
y_0 = [2.6,2,1,1,1.5,1.2,1.5,3]';
x_1 = [2.6,2.5,4.5,4.5,3.8,3.2,1.8,3,3.5]';
y_1 = [4.5,2.5,2,3,3.4,3.5,4,3.2,3]';
%散点图
plot(x_0,y_0,'ko');%原点是类别0
hold on
plot(x_1,y_1,'r^');%三角是类别1
axis([0,5,0,5]);
Y = [zeros(1,size(x_0,1)),ones(1,size(x_1,1))]';
X = [x_0,y_0;x_1,y_1];
x0 = ones(size(X,1),1);
X = [x0 X];
a = 0.00005;
theta = zeros(size(X,2),1);
new_theta = zeros(size(X,2),1);
d = 100;
%梯度上升法，找到极大似然参数
while (d>0.00001)
            new_theta(1) = theta(1) + a*(sum((Y-1./(1+exp(-X*theta))).*X(:,1)));
            new_theta(2) = theta(2) + a*(sum((Y-1./(1+exp(-X*theta))).*X(:,2)));
            new_theta(3) = theta(3) + a*(sum((Y-1./(1+exp(-X*theta))).*X(:,3)));
            dst = [theta';new_theta'  ];
            d = pdist(dst,'euclidean');
            theta = new_theta;
end
x =  0 : 0.1 : 5;
y = (-theta(2)*x - theta(1))/theta(3);
plot(x,y,'b-');
```
分类模型：
![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2017-12-30-AndrewNg-MachineLearning-lec3/fig2.JPG)

验证模型：
假设有样本$x_0(1,2.5,3.5)$,则$h_\theta(x)=0.9216$ 表明$x_0$有92.16%的概率属于类别1。而假设有样本$x_1(1,2.5,2.4)$，则$h_\theta(x)=0.4220$ 表明 $x_1$有58.8%的概率属于类别0。如果设阈值为50%，则可以清楚地进行分类。
但是这个模型是有缺陷的，对于$x(1,2.5,2.5)$这个样本点，分类器讲它认作是类别0的。因为这只是一个线性的决策边界，必然局限，如果采用非线性的决策边界，分类效果会更好。

#多类分类问题

多类分类问题其实只是双类分类问题的拓展，每次只要保留一类，剩下的为一类，即可找到该类所有的决策边界。
