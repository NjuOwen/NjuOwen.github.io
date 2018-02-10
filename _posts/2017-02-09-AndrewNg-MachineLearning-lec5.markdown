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
现用matlab中利用函数生成两组两维高斯分布的数据，再给定一测试点，利用高斯判别分析对测试点进行分类
![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2018-02-08-AndrewNg-MachineLearning-lec5/GDA.JPG)
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
GDA的算法的分界面是一个线性分界面$Ax=b(x\in R^n)$,其中
$$A=2\Sigma^{-1}(\mu_1-\mu_0)\\b=\mu_1^T\Sigma^{-1}\mu_1-\mu_0^T\Sigma^{-1}\mu_0+log\phi-log(1-\phi)$$
实际上GDA只是逻辑回归的一个特例。两者效果比较:
（1）逻辑回归是基于弱假设推导的，则其效果更稳定，使用范围更广
（2）当数据服从高斯分布时，GDA效果更好
（3）当训练样本数很大时，根据中心极限定理，数据将无限逼近于高斯分布，则此时GDA的表现效果会很好
#**朴素贝叶斯方法**
现在来讨论一个垃圾邮件分类器，一封邮件可以用一个向量$x\in R^n$表示，这个向量对应一个字典，如果字典$x_i$对应的词在邮件中出现，则对应的$x_i$为1，否则为0。假设$x_i$在给定$y$时是条件独立的，这个假设也是称为“朴素”的原因。则模型参数如下：
$$\phi_{i|y=1}=p(x_i|y=1)\\\phi_{i|y=0}=p(x_i|y=0)\\\phi_y=p(y=1)$$
作出极大似然估计：
$$\phi_{j|y=1}=\frac{\Sigma_{i=1}^{m}I \lbrace x_j^{(i)}=1,y^{(i)}=1 \rbrace}{\Sigma_{i=1}^{m}I \lbrace y^{(i)=1} \rbrace}\\ \phi_{j|y=0}=\frac{\Sigma_{i=1}^{m}I \lbrace x_j^{(i)}=0,y^{(i)}=0 \rbrace}{\Sigma_{i=0}^{m}I \lbrace y^{(i)=0} \rbrace}\\
\phi_y=\frac{\Sigma_{i=1}^{m}I\lbrace y^{(i)=1} \rbrace}{m} $$
那么分类器再收到一个新邮件$x$时，即可通过贝叶斯公式判断出该邮件是否是垃圾邮件。
$$y=arg\underset{y}max\ p(y|x)=arg\underset{y}max \ \frac{p(x|y)p(y)}{p(x)}=arg\underset{y}max\ p(x|y)p(y)\\p(x|y)=\Pi_{i=1}^{m}p(x_i|y=1)$$
#**拉普拉斯平滑**
如果新收到的邮件出现了一个新单词，为了防止计算出的概率为0，我们在分子分母上都加上一个常数，进行拉普拉斯平滑,其中K是类的个数。
$$p(y=c_k)=\frac{\Sigma_{i=1}^{m}I \lbrace y_i=c_k \rbrace+\lambda}{m+K\lambda}$$
对应于朴素贝叶斯方法中的修正是
$$\phi_{j|y=1}=\frac{\Sigma_{i=1}^{m}I \lbrace x_j^{(i)}=1,y^{(i)}=1 \rbrace+1}{\Sigma_{i=1}^{m}I \lbrace y^{(i)=1} \rbrace+2}$$
