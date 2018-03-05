---
layout: post
title:  "AndrewNg-MachineLearning-Lecture12"
subtitle: “K-means聚类 EM算法 Jensen不等式”
date:   2018-03-02
categories: [Clustering,K-means,Expertation Maximitation,Jensen's inquality]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**K-means聚类**

之前讨论的都是监督学习，聚类算法则是一种无监督学习，原始数据集中是没有给定标签的，通过聚类，将近似的样本划为一类，并给出样本的标签。K-means是聚类算法中的一种，步骤如下:首先初始化中心点，欲划分几类就有几个中心点。接下来:i)设置样本$i$的标签$c^{(i)}=argmin_j||x^{(i)}-\mu_j||$,以距离最近的中心点作为自己的类。ii)更新所有中心点$\mu_j=\frac{\Sigma_i=1^mI\lbrace c^{(i)}=j \rbrace x^{(i)}}{\Sigma_i=1^mI\lbrace c^{(i)}=j \rbrace}$，重复上述步骤直到收敛。重复步骤一定会收敛吗？可以看一下代价函数$J(c,\mu)=\Sigma_{i=1}^m||x^{(i)}-\mu_c^{(i)}||^2$，K-means算法实际上是对于$J$的坐标上升算法。
西瓜数据集4.0如下：
![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2018-03-02-AndrewNg-MachineLearning-lec12/data1.JPG)
通过K-means聚类结果如下(分成3类，聚类中心用黑色六角星表示):
![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2018-03-02-AndrewNg-MachineLearning-lec12/Kmeans.JPG)
matlab代码如下:
```matlab
clc,clear
%西瓜数据集4.0
%属性1 密度
x1=[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719,0.359,0.339,0.282,0.748,0.714,0.483,0.478,0.525,0.751,0.532,0.473,0.725,0.446];
%属性2 含糖率
x2=[0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103,0.188,0.241,0.257,0.232,0.346,0.312,0.437,0.369,0.489,0.472,0.376,0.445,0.459];
%plot(x1,x2,'ko');%原始无标签数据绘图
%假设k=3有3类
k=3;
mu=zeros(2,3);
%初始化中心点
for i = 1 : k
    mu(1,i)=x1(1,i*10);
    mu(2,i)=x1(1,i*10);
end
l = size(x1,2);
label=zeros(1,l);
temp_label= zeros(1,2);
d=1;
while(d~=0)
    for i = 1 : l
        x=[x1(i);x2(i)];%样本点
        temp= [norm(mu(:,1)-x),norm(mu(:,2)-x),norm(mu(:,3)-x)];
        label(i) = find(temp==min(temp));
    end
    %更新中心点
    num_mu=zeros(1,3);
    temp_mu1=zeros(2,1);
    temp_mu2=zeros(2,1);
    temp_mu3=zeros(2,1);
    for i = 1 : l
        if label(i)==1
            num_mu(1)=num_mu(1)+1;
            temp_mu1=[x1(i);x2(i)]+temp_mu1;
        elseif label(i)==2
            num_mu(2)=num_mu(2)+1;
            temp_mu2=[x1(i);x2(i)]+temp_mu2;
        else
            num_mu(3)=num_mu(3)+1;
            temp_mu3=[x1(i);x2(i)]+temp_mu3;
        end
    end
    d=norm(temp_mu1/num_mu(1)-mu(:,1))+norm(temp_mu2/num_mu(2)-mu(:,2))+norm(temp_mu3/num_mu(3)-mu(:,3));
    mu(:,1)=temp_mu1/num_mu(1);
    mu(:,2)=temp_mu2/num_mu(2);
    mu(:,3)=temp_mu3/num_mu(3);
end
%绘出结果
for i = 1 : l
    if label(i)==1
        plot(x1(i),x2(i),'bo','MarkerFaceColor','b');
        hold on
    elseif label(i)==2
        plot(x1(i),x2(i),'ro','MarkerFaceColor','r');
        hold on
    else
        plot(x1(i),x2(i),'go','MarkerFaceColor','g');
        hold on
    end
end
%绘出中心点
plot(mu(1,:),mu(2,:),'kh');
xlabel('密度');
ylabel('含糖率');
```

#**Jensen不等式**

为了解决之后的EM的问题，这里先介绍Jensen不等式。对于凸函数$f(x)$(即$f''(x)\geq0$)和随机变量$X$,则有$f(EX)\leq E[f(x)]$。而如果$f(x)$是严格凸函数(即f''(x)>0)，那么当且仅当$X=E[X]$时，$E[F(X)]=f(EX)$
而如果$f(x)$是凹函数,那么不等号方向反向即可,$E[f(X)]\leq f(EX)$。

#**EM算法**

如果有一个模型$p(x,z;\theta)$,但是给定的训练样本只有$\lbrace x^{(1)},\cdots,x^{m} \rbrace$,变量$z$是隐形的无法观察得到。它的极大似然估计函数是$l(\theta)=\Sigma_{i=1}^m logp(x^{(i)};\theta)=\Sigma_{i=1}^m log\Sigma_{z^{(i)}}p(x^{(i)},z^{(i)};\theta)$,因为有隐藏变量$z$的存在,直接求$\theta$比较困难。
EM算法是一种解决存在隐含变量优化问题的有效方法,它通过不断优化$l(\theta)$的下界(M步),再优化下界(M步)，最终得到$l(\theta)$的最大值。
现在假设$Q_i$表示隐含变量$z$的某种分布,且$Q_i$满足条件$\Sigma_{z^{(i)}}Q_i(z^{(i)})=1,Q_i(z^{(i)})\geq0$,可见$Q_i$是概率密度函数。那么可以改写$l(\theta)$,并利用Jensen不等式构造下界。
$$
l(\theta)=\Sigma_i log\Sigma_{z^{(i)}}p(x^{(i)},z^{(i)};\theta)=\Sigma_i log\Sigma_{z^{(i)}}Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\geq\Sigma_i \Sigma_{z^{(i)}}Q_i(z^{(i)})log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$
我们希望下界尽量紧贴$l\theta$,我们可以先使不等式取等，再优化$\theta$取得下界的最大值，再以更新的$\theta$使不等式取等，直到收敛，取得$l(\theta)的最大值$。Jensen不等式取等的条件是随机变量为常数，即$\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}=c$而又有条件$\Sigma_{z^{(i)}}Q_i(z^{(i)})=1$，可以得到$\Sigma_{z^{(i)}}p(x^{(i)},z^{(i)};\theta)=c$。那么改写不等式取等条件$Q_i(z^{(i)})=\frac{p(x^{(i)},z^{(i)};\theta)}{\Sigma_{z^{(i)}}p(x^{(i)},z^{(i)};\theta)}=\frac{p(x^{(i)},z^{(i)};\theta)}{p(x^{(i)};\theta)}=p(z^{(i)}|x^{(i)};\theta)$可以得出Q_i就是后验概率。
所以EM算法如下：
E步：对于每一个$i$计算$Q_i(z^{(i)})=p(z^{(i)}|x^{(i)};\theta)$
M步：计算$\theta:=argmax_{\theta}\Sigma_i \Sigma_{z^{(i)}}Q_i(z^{(i)})log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}$
重复E步和M步直到收敛。这也可以看作是坐标上升法，E步固化$\theta$优化$Q_i$,M步固化$Q_i$优化$\theta$。
