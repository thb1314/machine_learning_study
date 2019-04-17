#!/usr/bin/python
# -*- coding: UTF-8 -*-
from Tkinter import Image

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
print(grid.best_score_, grid.best_params_)


