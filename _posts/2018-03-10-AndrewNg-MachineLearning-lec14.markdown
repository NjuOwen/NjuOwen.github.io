---
layout: post
title:  "AndrewNg-MachineLearning-Lecture14"
subtitle: “主成分分析”
date:   2018-03-10
categories: [Principle Component Analysis]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**主成分分析**

对于一个数据集$D$，有时样本维度极高远远大于给定样本的个数，有时样本的一些特征之间存在依赖关系，部分信息是冗余的实际并不需要那么多维特征去描述，且高维特征消耗大量的计算资源，也不便于数据的可视。这时我们希望对样本降维，PCA(主成分分析)便是其中应用最广泛的一种降维算法。
给定样本$\lbrace x^{(1)},\dots,x^{(m)} \rbrace$,希望将它降维到$k$维。首先对给定样本进行白化:(1)设置样本均值$\mu=\frac{1}{m}\Sigma_{i=1}^{m}x^{(i)} $(2)每个样本$x^{(i)}$都减去均值\mu,使均值为0(3)设置样本方差$\sigma^2=\frac{1}{m}\Sigma_{i=1}^m(x_j^{(i)})^2 $(4)每个样本的每个维度$x_j^{(i)}$都除以响应的方差$\sigma_j$统一各特征量纲。
降维需要基变换，那么优化目标就是如何选取基，使其能保留最多的信息。可以想象，数据集在某个基上的投影值越分散，方差越大，这个基所保留的信息也就越多。PCA正是采用方差来度量信息量的。优化目标可以写作$max_\mu\frac{1}{m}\Sigma_{i=1}^m ({x^{(i)}}^T u-\mu)^2$,因为已经做过白化处理，此时$\mu=0$,故优化问题为$max_\mu \frac{1}{m}\Sigma_{i=1}^m ({x^{(i)}}^T u)^2=\frac{1}{m}\Sigma_{i=1}^m({x^{(i)}}^T u)(u^T x^{(i)})=u^T [\frac{1}{m}\Sigma_{i=1}^m(x^{(i)}{x^{(i)}}^T)]u$。其中$\frac{1}{m}\Sigma_{i=1}^m(x^{(i)}{x^{(i)}}^T)$正是原数据集$D$的协方差矩阵,用$C$表示。那么在该基底下保留的信息量$info(D,u)=u^T C u$,加入限制$||u||=1$使基底是单位向量，利用拉格朗日乘数法构造$f(u)=u^T C u-\lambda(u^T u-1)$,求导得$Cu=\lambda u$,所以信息保存能力最大的基底$u$正是协方差矩阵$C$的特征向量。再回去看$info(D,u)=u^T \lambda u=\lambda u^T u=\lambda$，可以发现基底对应的特征值越大，保存的信息量也就越大。那么虽然协方差矩阵可能有若干特征向量作为基底，但是实际上只要选取特征值最大的几个特征向量作为基底就可以保留原数据集的绝大部分信息。
下面通过鸢尾花的数据进行主成分分析。
matlab代码如下:
```matlab
clc,clear
%PCA主成分分析
%载入数据
filename = 'Irisdata.txt';
[x1,x2,x3,x4,x5] = textread(filename,'%f%f%f%f%s','delimiter',',');
data = [x1,x2,x3,x4]';
spec = x5;
%数据预处理
data = data-repmat(mean(data,2),[1 size(data,2)]);%均值0
data = data./repmat(sqrt(mean(data.^2,2)),[1 size(data,2)]);%归一化
%计算协方差矩阵
Sigma = zeros(4,4);
for i = 1 : size(data,2)
    Sigma = Sigma + data(:,i)*data(:,i)';
end
Sigma = Sigma/size(data,2);
[w,~] = eigs(Sigma,[],3);%提取特征值较大的前三个特征向量
PCA_data = [w(:,1)'*data;w(:,2)'*data;w(:,3)'*data];
%绘制PCA散点图
for i = 1 : size(data,2)
    if strcmp(x5(i),'Iris-setosa')
        scatter3(PCA_data(1,i),PCA_data(2,i),PCA_data(3,i),'b','filled');
        hold on
    elseif strcmp(x5(i),'Iris-versicolor')
        scatter3(PCA_data(1,i),PCA_data(2,i),PCA_data(3,i),'y','filled');
        hold on
    else
        scatter3(PCA_data(1,i),PCA_data(2,i),PCA_data(3,i),'g','filled');
        hold on
    end
end
xlabel('PC1');ylabel('PC2');zlabel('PC3');
```
当选取最大的两个特征值对应的特征向量作基底时，可以保留原数据集97.76%的信息

当选取最大的两个特征值对应的特征向量作基底时，可以保留原数据集99.48%的信息
