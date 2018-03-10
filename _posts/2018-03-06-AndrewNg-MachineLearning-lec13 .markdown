---
layout: post
title:  "AndrewNg-MachineLearning-Lecture13"
subtitle: “混合高斯模型 混合朴素贝叶斯模型”
date:   2018-03-06
categories: [Mixture of Gaussian,Mixture of Naive Bayes]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**混合高斯模型**

高斯混合模型(MoG)是运用EM算法解决隐含变量优化问题的一个例子。给定训练样本$\lbrace x^{(1)},\cdots,x^{(m)}\rbrace$,隐含类别标签用$z^{(i)}$表示，并且$z^{(i)}$服从多项式分布，即$z^{(i)}\sim Multinoumial(\phi)$($\phi_j\geq0$,$\Sigma_j\phi_j=1$)。在给定类别标签下随机变量$x^{(i)}$服从高斯分布，即$x^{(i)}|z^{(i)}=j\sim N(\mu_j,\Sigma_j)$。下面运用EM算法，循环直至收敛:
E步:对于每一个样本$i$和每个样本的每个可能类别$j$，计算$w_j^{(i)}=p(z^{(i)}=j|x^{(i)};\phi,\mu,\Sigma)$,由全概率公式和贝叶斯公式展开得$w_j^{(i)}=\frac{p(x^{(i)}|z^{(i)}=j)p(z^{(i)}=j)}{\Sigma_{l=1}p(x^{(i)}|z^{(i)}=l)p(z^{(i)}=l)}$
M步:更新参数
$$
\phi_j=\frac{1}{m}\Sigma_{i=1}^{m}w_j^{(i)}\\
\mu_j=\frac{\Sigma_{i=1}^{m}w_j^{(i)}x^{(i)}}{\Sigma_{i=1}^{m}w_j^{(i)}}\\
\Sigma_j=\frac{\Sigma_{i=1}^{m}w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\Sigma_{i=1}^{m}w_j^{(i)}}
$$
与K-means不同，EM算法使用了"软"指定，对每个样本的每个类别标签赋予概率，而非"硬"指定标签,相应地计算量也变大了。
matlab代码如下:
```matlab
clc,clear
%Mixture of Gaussian 高斯混合模型
%数据准备
N=10000;%样本个数
phi1=0.1;mu1 = [1 10];sigma1 = [1 0; 0 2];
phi2=0.3;mu2 = [3 6];sigma2 = [1 0; 0 3];
phi3=0.6;mu3 = [7 3];sigma3 = [2 0; 0 2];
x=zeros(N,2);
%生成多维正态数据
for i = 1 : N
    rate = rand;
    if rate <= phi1
        x(i,:) = mvnrnd(mu1,sigma1,1);
    elseif rate <= phi1+phi2
        x(i,:) = mvnrnd(mu2,sigma2,1);
    else
        x(i,:) = mvnrnd(mu3,sigma3,1);
    end
end
subplot(1,2,1);plot(x(:,1),x(:,2),'b+');hold on
%初始化
mu=[0 12;5 5;10 2];
sigma=[1 0;0 1;1 0;0 1;1 0;0 1];
phi = [0.33, 0.33, 0.34];
w = zeros(N,3);
T = 100;%设置迭代次数
for i = 1 : T
    % Expectation
    for k = 1 : 3
        w(:,k) = phi(k)*mvnpdf(x,mu(k,:),sigma((2*k-1):2*k,:));%
    end
    w = w./repmat(sum(w,2),[1 3]);
    % Maximization
    for k = 1 : 3
        mu(k,:) = w(:,k)'*x / sum(w(:,k));
        temp_m=zeros(2,2);
        for j = 1 : N
            temp=(x(j,:)-mu(k,:))'*(x(j,:)-mu(k,:));
            temp_m=temp_m+temp*w(j,k);
        end
        sigma((2*k-1):2*k,:) =temp_m/sum(w(:,k));
        phi(k) = sum(w(:,k)) / N;
    end
end
%画出MoG分类后的类别图
subplot(1,2,2);
for i = 1 : N
color_m =[mvnpdf(x(i,:),mu(1,:),sigma(1:2,:)) mvnpdf(x(i,:),mu(2,:),sigma(3:4,:)) mvnpdf(x(i,:),mu(3,:),sigma(5:6,:))];
color = find(color_m==max(color_m));
switch color
    case 1
        plot(x(i,1),x(i,2),'ro');
        hold on
    case 2
        plot(x(i,1),x(i,2),'bo');
        hold on
    otherwise
        plot(x(i,1),x(i,2),'go');
        hold on
end
end
```
最终将无标签的样本空间，按照指定的类别数分类。
![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2018-03-06-AndrewNg-MachineLearning-lec13/EM_2d.JPG)
