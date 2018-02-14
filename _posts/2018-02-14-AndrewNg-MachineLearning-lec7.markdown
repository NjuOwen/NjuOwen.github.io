---
layout: post
title:  "AndrewNg-MachineLearning-Lecture7"
subtitle: “最优间隔分类器 对偶问题 支持向量机”
date:   2018-02-14
categories: [Optimal Margin Classifier,Dual Optimization,SVM]

---
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

#**最优间隔分类器**

对于优化问题:
$$
max_{\hat{\gamma},w,b} \frac{\hat{\gamma}}{||w||}\ ( \frac{\hat{\gamma}}{||w||}=\gamma)\\
s.t.\ y^{(i)}\cdot(w^Tx^{(i)}+b)\geq\hat{\gamma}\\
$$
加入约束，令函数间隔$\hat{\gamma}=1$即$min_{i}\ y^{(i)}\cdot(w^Tx^{(i)}+b)=1$,于是优化问题可以改写为:
$$
min_{w,b}\ ||w||^2\\
s.t.\ y^{(i)}(w^Tx^{(i)}+b)\geq1\\
$$
这就构成了一个最优间隔分类器

#**对偶优化**

为了求解最优间隔分类器，需要对模型进行对偶优化。对于**原问题**:
$$
min\ f(x)\\
s.t.\  g_i(x)\leq0,\ i=1,\cdots,k\\
h_i(x)=0,\ i=1,\cdots,l
$$
它的拉格朗日对偶函数为
$$
\theta(\lambda,\mu)=i\underset{x}nf\ L(x,\lambda,\mu)=i\underset{x}nf(f(x)+\Sigma_{i=1}^m\lambda_ig_i(x)+\Sigma_{i=1}^p\mu_ih_i(x))
$$
规定其中$\lambda_i\geq0$显然$\theta(\lambda,\mu)\leq p^* $,其中$ p^* $是原问题的最优解
原问题的**拉格朗日对偶问题**如下:
$$
max\ \ \theta(\lambda,\mu)\\
s.t.\  \lambda_i\geq0\\
 \Sigma_{i=1}^m\lambda_i^* g_i(x^* )=0
$$
那么对偶问题的最优值$d^* $就是$p^* $的最优下界，即所有下界中离$p^* $最近的一个，并且$d^* \leq p^* $,我们将这个等式称作弱对偶性质，定义$p^*-d^*$为对偶间隙,现在希望对偶间隙等于0，以此解决对偶问题即可解决原问题。事实上当满足Slater条件时，对偶间隙为0。对偶条件如下:
$$
\exists x\ \ s.t.\\
 g_i(x)<0,\ i=1,\cdots,k\\
 h_i(x)=0,\ i=1,\cdots,l
$$
当满足Slater条件时，$d^* = p^* $,这种情况称为强对偶性质。此时，对偶问题的最优点$\lambda^* ,\mu^* $对应的最优值$d^* =\theta(\lambda^* ,\mu^* )=p^* $。
$$
\theta(\lambda^* ,\mu^* )=i\underset{x}nf\ L(x,\lambda^* ,\mu^* )=f(x^* )+\Sigma_{i=1}^m\lambda_i^* g_i(x^* )+\Sigma_{i=1}^p\mu_i^* h_i(x^* )\overset{(1)}=f(x^* )=p^*
$$
从等号(1)可以看出$\Sigma_{i=1}^m\lambda_i^* g_i(x^* )=0$因为规定$\lambda_i\geq0$，所以有$\forall i,\lambda_i^* g_i(x^* )=0$这被称为“互补性条件”。也就是说$\lambda_i^* $和$g_i(x^* )$每次必有一个为0。所以只有积极约束$(g_i(x^* )=0)$才可以得到不为0的对偶变量$\lambda$。

#**支持向量机**

支持向量机寻找超平面可以归结为一个优化问题(对最优间隔分类器的优化问题作适当变形)
$$
min \ \frac{1}{2}||w||^2=\frac{1}{2}w^Tw\\
s.t.\ -y^{(i)}(w^Tx^{(i)}+b)+1\leq0
$$
由之前的对偶优化的推导，只有当$g_i(w)=-y^{(i)}(w^Tx^{(i)}+b)+1=0$是积极约束的时候，这个约束对应的对偶变量才不会$0$。$g_i(w)=0$也就是说函数间隔等于1。函数间隔为1的样本点被称为支持向量，所以只有支持向量才对应不为0的拉格朗日乘子。这也是支持向量机名字的由来。
将对偶优化运用到支持向量机中
$$
\theta(w,b,\alpha)=i\underset{w}nf(\frac{1}{2}w^Tw-\Sigma_{i=1}^{m}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1))\\
令L(w,b,\alpha)=\frac{1}{2}w^Tw-\Sigma_{i=1}^{m}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)
$$
为了找到$\theta$的最小值，求导

令$\nabla_wL(w,b,\alpha)=0$，得$w^* =\Sigma_{i=1}^m\alpha_iy^{(i)}x^{(i)}$,将$w$代入$L$，得
$$\theta=\Sigma_{i=1}^m\alpha_i-\frac{1}{2}\Sigma_{i=1}^m\Sigma_{j=1}^my^{(i)}y^{(j)}\alpha_i\alpha_j<x^{(i)}x^{(j)}>
$$

令$\frac{d}{db}L(w,b,\alpha)=\Sigma_{i=1}^m\alpha_iy^{(i)}=0$

那么现在得优化问题是:
$$
max\ \ \theta(\alpha)\\
s.t.\  \alpha_i\geq0\\
\Sigma_{i=1}^m\alpha_iy^{(i)}=0
$$
只要求出$\alpha_i$即可求出$w^* (=w)$再根据$b^* =\frac{max_{i:y^{(i)}=-1}w^Tx^{(i)}+min_{i:y^{(i)}=1}w^Tx^{(i)}}{2}=b$
得到超平面分界面$(w,b)$。
为什么要引入对偶问题呢，现在来看超平面
$$
w^Tx+b=(\Sigma_{i=1}^m\alpha_iy^{(i)}x^{(i)})^Tx+b=\Sigma_{i=1}^m\alpha_iy^{(i)}<x^{(i)},x>+b
$$
对于$w^Tx+b$需要将待分类样本与$w,b$作线性计算，而对于$\Sigma_{i=1}^m\alpha_iy^{(i)}<x^{(i)},x>+b$
只需将待分类样本与支持向量作内积
因为非支持向量对应的$\alpha_i=0$。利用对偶优化，大大提高了分类的效率。
