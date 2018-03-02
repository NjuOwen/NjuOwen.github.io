---
layout: post
title:  "AndrewNg-MachineLearning-Lecture8"
subtitle: “K-means聚类 EM算法 Jensen不等式”
date:   2018-03-02
categories: [Clustering,K-means,Expertation Maximitation,Jensen's inquality]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**K-means聚类**

之前讨论的都是监督学习，聚类算法则是一种无监督学习，原始数据集中是没有给定标签的，通过聚类，将近似的样本划为一类，并给出样本的标签。K-means是聚类算法中的一种，步骤如下:首先初始化中心点，欲划分几类就有几个中心点。接下来:i)设置样本$i$的标签$c^{(i)}=argmin_j||x^{(i)}-\mu_j||$,以距离最近的中心点作为自己的类。ii)更新所有中心点$\mu_j=\frac{\Sigma_i=1^mI\lbrace c^{(i)}=j \rbrace x^{(i)}}{\Sigma_i=1^mI\lbrace c^{(i)}=j \rbrace}$，重复上述步骤直到收敛。重复步骤一定会收敛吗？可以看一下代价函数$J(c,\mu)=\Sigma_{i=1}^m||x^{(i)}-\mu_c^{(i)}||^2$，K-means算法实际上是对于$J$的坐标上升算法。
西瓜数据集4.0如下：
