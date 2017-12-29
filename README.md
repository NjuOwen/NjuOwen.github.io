[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/kronik3r/daktilo/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

# Daktilo
Daktilo is a [Jekyll](jekyllrb.com) theme with a minimal design inspired from typewriters.

# More info and Live preview
[Click here](http://daktilo.github.io/) to see the theme in action.

# Features
- Fully responsive
- [Disqus](https://disqus.com/) integration for comments.
- Google Analytics integration.
- Syntax Highlighter (using [highlight.js](https://highlightjs.org/)).
- Support for categories.
- Font-Awesome Icons.
- Optimized for SEO.
- Coolest [404 page ever](http://electrik-frog.com/daktilo/404.html).

# How to use it
Start by cloning the repository, then check the `_config.yml` file and change it accordingly.
Note that the `title` property is what will be displayed as logo.

Finally execute `jekyll serve --watch` and head to [localhost:4000](http://127.0.0.1:4000) to see the result.

# Using categories
Categories are little bit tricky. Please make sure to do the following for each category:

- Create a file within `categories` folder with the name of your category
For example let's say that we have a category called `An Awesome Category` you will need to add an `an-awesome-category.html` file with this content:

``` html
---
layout: category
category: an-awesome-category
permalink: /categories/an-awesome-category/
---

```

- Create an entry inside `_data/categories.yml`

``` html
- slug: an-awesome-category
  name: An Awesome Category
```

- Then you will see it in the footer in the `Explore` section.

# License

The contents of this repository is licensed under [The MIT License.](https://opensource.org/licenses/MIT)

![](C:/Users/文昊/Documents/GitHub/NjuOwen.github.io/img/2017-12-29-AndrewNg-MachineLearning-lec2/正规方程效果图.JPG)
![](C:/Users/文昊/Documents/GitHub/NjuOwen.github.io/img/2017-12-29-AndrewNg-MachineLearning-lec2/单自变量梯度下降.JPG)
![](C:/Users/文昊/Documents/GitHub/NjuOwen.github.io/img/2017-12-29-AndrewNg-MachineLearning-lec2/多变量梯度下降_散点图.JPG)
![](C:/Users/文昊/Documents/GitHub/NjuOwen.github.io/img/2017-12-29-AndrewNg-MachineLearning-lec2/多自变量梯度下降.JPG)