---
layout: post
title:  "AndrewNg-MachineLearning-Lecture3 "
subtitle: “局部加权回归 逻辑回归 多元分类”
date:   2017-12-30
categories: [machine leaning,locally weighted regression,logistic regression,classification]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#局部加权回归

现实生活中很多情况并不可以作简单的一次线性回归，有时必须用高次线性回归，但在选取次数上十分麻烦，次数选取不好就会出现欠拟合或是过拟合的情况。为了规避这麻烦，可以采用局部加权回归。通过权值函数$W^{(i)}=e^{-\frac{(x^{(i)}-x)^2}{2\tau^2}}$(钟形函数)对代价函数$J(\theta)$进行修正，削弱远离预测点的样本点对回归结果的影响。
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
在LWR方法中，$\tau$的选取对结果至关重要。


预测模型：
![](https://raw.githubusercontent.com/NjuOwen/NjuOwen.github.io/master/img/2017-12-29-AndrewNg-MachineLearning-lec2/fig4.JPG)
