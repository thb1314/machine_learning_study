<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: { inlineMath: [['$','$'], ['$ ',' $']], processClass: 'math', processEscapes: true },
        'HTML-CSS': { linebreaks: { automatic: true } },
        SVG: { linebreaks: { automatic: true } }
        });
</script>
<script src="https://mathjax.cnblogs.com/2_7_2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# 第二周机器学习


>     本文公式显示需要使用Mathjax，然后令人悲伤的是github不支持Mathjax
>     您可以将这篇md文件pull下来，使用您本地的markdown解析器解析
>     没有必要在公示显示上浪费时间，您也可以下载我本地生成的html用浏览器打开即可
>     或者您也可以下载我上传到github上的pdf
*[Mathjax开源项目地址](https://github.com/mathjax/MathJax)*


## Test and Debug Your ML System

### Debug the ML System
待补充
### Machine learning diagnostic
待补充
### Evalating your hypothesis
待补充


## 决策树

> 这一章没有看吴恩达老师的视频，看的是中科院某博士讲的


### Code
> [参考博客链接](https://www.cnblogs.com/pinard/p/6056319.html)

```Python
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
# 这里我们使用scikit-learn(sklearn)作为我们机器学习的模块
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

house_data = fetch_california_housing()
# print(house_data.DESCR)

# 定义树的最大深度等于2
dtr = tree.DecisionTreeRegressor(max_depth=2)
dtr.fit(house_data.data[:, [6, 7]], house_data.target)


dot_data = \
    tree.export_graphviz(dtr,
                         out_file=None,
                         feature_names=house_data.feature_names[6:8],
                         filled=True,
                         impurity=False,
                         rounded=True
                         )
#
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor('#FFF2DD')
img_data = graph.create_png()
plt.imshow(plt.imread(BytesIO(img_data)))
# plt.show()
# graph.write_png('dtr_white_background.png')

# 拆分训练集 测试集
# 取0.1作为测试集
data_train, data_test, target_train,target_test = \
    train_test_split(house_data.data, house_data.target, test_size = 0.1,random_state = 42)
dtr = tree.DecisionTreeRegressor(random_state = 42)
dtr.fit(data_train, target_train)
print(dtr.score(data_test, target_test))


# 使用GridSearchCV帮助我们选择合适参数
tree_param_grid = {'min_samples_split': list((3, 6, 9)), 'n_estimators': list((10, 50, 100))}
grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print(grid.grid_scores_, grid.best_score_, grid.best_params_)

```

运行结果

```
0.637355881715626
(0.8074196516933743, {'min_samples_split': 6, 'n_estimators': 100})
```


## (Ensemble learning)集成算法
>	Note：集成算法并不是机器学习算法的一种，而相当于把很多个机器学习算法拢在一块。

### Ensemble learning介绍
- Ensemble learning
  - 目的：让机器学习效果更好，单个不行，那就一群
  - Bagging: 训练多个分类器取平均：$ f(x) = \frac{1}{M} \sum_{m=1}^M f_m(x)​$
  - Boosting：从弱学习器开始加强，通过加权来进行训练
    - $ F_m(x) = F_{m-1}(x) + argmin_h\sum_{i=1}^n L(y_i,F_{m-1}(x_i)+h(x_i))  ​$ 加入一个数，要比原来强
  - Stacking:聚合多个分类和回归模型（可以分阶段来做)
### Bagging模型
- 全称：bootstrap aggregation(说白了就是并行训练一堆分类器)
- 最典型的代表就是随机森林
- 随机：数据采样随机(一般取60%-80%，有放回)，特征选择随机（获得一系列随机的树以后对当中特征也按照60%-80%进行采样）
- 之所以要进行随机，是要保证泛化能力
- 森林：很多决策树并行放在一起
- 理论上越多的树效果会越好，但实际上基本超过一定数量就差不多上下浮动了

![深度截图_选择区域_20190416122403](../../../../桌面/深度截图_选择区域_20190416122403.png)

#### 随机森林优势
- 可以处理很多维度(feature很多)的数据，并且不用做特征选择
- 在训练完后，它能够给出哪些feature比较重要
- 容易做成并行化方法，速度比较快
- 可以进行可视化展示，便于分析

### Boosting模型
- 典型代表：AdaBoost，Xgboost
- Adaboost会根据前一次的分类效果调整数据权重
- 解释：如果某一个数据在这次分错了，那么下一次就会给他更大的权重
- 结果：每个分类器根据自身准确性来确定各自的权重，再合体

### Stacking模型
- 堆叠：很暴力，拿来一堆直接上
- 可以堆叠各种各样的分类器（KNN、SVM、RF等）
- 分阶段：第一阶段得出各自结果，第二阶段再用前一阶段结果训练
- 为了刷结果，不择手段

堆叠在一起确实能使得准确率得到提升，但是速度是个问题  
集成算法是竞赛与论文神奇，当我们更关注与结果时不妨来试试！














