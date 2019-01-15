from IPython.display import Image  
import pandas as pd
import pydot
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
dtree = DecisionTreeClassifier()
dtree.fit(df, y)
dot_data = export_graphviz(dtree, out_file=None,
                filled=True, rounded=True,
                special_characters=True)
(graph,) = pydot.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('dtree.png')
