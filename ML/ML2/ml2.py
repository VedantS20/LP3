import pydotplus
from sklearn.tree import export_graphviz
from IPython.display import Image
from six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# reading Dataset   
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 5]

# Perform Label encoding
le = LabelEncoder()

X = X.apply(le.fit_transform)
print(X)

regressor = DecisionTreeClassifier(criterion="entropy")
regressor.fit(X.iloc[:, 1:5], y)

# Predict value for the given Expression
X_in = np.array([1, 1, 0, 0])
y_pred = regressor.predict([X_in])
print("Prediction:", y_pred)

dot_data = StringIO()

# dtree = dtreeplt(model=regressor,feature_names=['Age',"Income","Gender","Marital Status"],target_names = np.array(['Yes','No']))

# fig = dtree.view()

# fig.savefig('op.png')

export_graphviz(regressor, out_file=dot_data, filled=True,
                rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree1.png')

