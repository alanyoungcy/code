### 回归算法器
```python
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDRegressor, HuberRegressor, LinearRegression
from sklearn.linear_model import PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor

```


### 分类算法器
```python
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

```


### 网格搜素
```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid,
                               scoring="accuracy", return_train_score=True)

```