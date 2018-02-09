---
layout: post
title:  "AndrewNg-MachineLearning-Lecture5 "
subtitle: “高斯判别分析 朴素贝叶斯算法 拉普拉斯平滑”
date:   2018-02-09
categories: [Gaussian Discriminant Analysis,Naive Bayes,Laplace smoothing]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**高斯判别方法**
高斯判别分析是一种分类算法。对于两类样本，其服从伯努利分布且，而对每个类中的样本，假定其服从高斯分布则有
$$ y\sim Bernouli(\phi)\\x|y=0\sim N(\mu_0,\Sigma)\\x|y=1\sim N(\mu_1,\Sigma)$$
对未知量$\phi,\mu_0,\mu_1,\Sigma$进行极大似然估计
$$\phi=\frac{I\lbrace y^{(i)}=1 \rbrace}{m}\\\mu_0=\frac{\Sigma_{i=1}^m (I\lbrace y^{(i)}=0\rbrace x^{(i)})}{\Sigma_{i=1}^m (I\lbrace y^{(i)}=0\rbrace}\\\mu_1=\frac{\Sigma_{i=1}^m (I\lbrace y^{(i)}=1\rbrace x^{(i)})}{\Sigma_{i=1}^m (I\lbrace y^{(i)}=1\rbrace}\\\Sigma=\frac{1}{m}\Sigma_{i=1}^{m}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T$$
这样再根据训练样本，估计出先验概率以及高斯分布的均值和协方差矩阵，即可通过贝叶斯公式求出一个新样本分别属于两类的概率，通过比较取其大着，进而实现对样本的分类。
$$p(y|x)=\frac{p(x|y)p(y)}{p(x)}\\N(x|\mu,\Sigma)=\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{|\Sigma|^{\frac{1}{2}}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\\y=arg\underset{y}max\ p(y|x)=arg\underset{y}max \ \frac{p(x|y)p(y)}{p(x)}=arg\underset{y}max\ p(x|y)p(y)$$
matlab代码如下:
```matlab
clc,clear
%Gaussian Discriminant Analysis 高斯判别分析
%数据准备
mu = [2 3];
SIGMA = [1 0; 0 2];
data0 = (mvnrnd(mu,SIGMA,100))';%生成多维正态数据
mu = [7 8];
SIGMA = [ 1 0; 0 2];
data1 = (mvnrnd(mu,SIGMA,100))';
plot(data1(:,1),data1(:,2),'*');
label0 = zeros(1,size(data0,2));
label1 = zeros(1,size(data1,2));
label = [label0,label1];
data = [label ; data0,data1];
%似然估计
L0 = size(data0,2);
L1 = size(data1,2);
phi = L1/(L0 + L1);
u0 = (sum(data0')/L0)';
u1 = (sum(data1')/L1)';
sigma = zeros(2,2);
for i = 1 : (L0+L1)
    if i <= L0
        new = (data(2:3,i)-u0)*(data(2:3,i)-u0)';     
    else
        new = (data(2:3,i)-u1)*(data(2:3,i)-u1)';     
    end
    sigma = sigma + new;
end
%测试点test_x=4 test_y=3 truth=1
test = [4;4];
D = size(data,1)-1;
p0 = (1/(2*pi)^(D/2))*(1/det(sigma)^0.5)*exp(-0.5*(test-u0)'*inv(sigma)*(test-u0));
p1 = (1/(2*pi)^(D/2))*(1/det(sigma)^0.5)*exp(-0.5*(test-u1)'*inv(sigma)*(test-u1));
P0 = p0*(1-phi);
P1 = p1*phi;
if(P0>P1)
    disp('测试点属于0类');
else
    disp('测试点属于1类');
end
%散点图
plot(data0(1,:),data0(2,:),'r+');
hold on;
plot(data1(1,:),data1(2,:),'*');
plot(test(1),test(2),'bh');

```
