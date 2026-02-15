# data analysis
import pandas as pd
import numpy as np

# for rendering tables in jupyter
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# feature tables
train_df = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/train.csv')
val_df = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/val.csv')
test_df = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/test.csv')

train_df = train_df.set_index('original_SampleID')
val_df = val_df.set_index('original_SampleID')
test_df = test_df.set_index('original_SampleID')

X_train = train_df.iloc[:, :-1]
y_train = train_df['total']

X_val = val_df.iloc[:, :-1]
y_val = val_df['total']

X_test = test_df.iloc[:, :-1]
y_test = test_df['total']

from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])

# Configure Cross-Validation
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

# Fit LassoCV
model = make_pipeline(
    StandardScaler(),
    LassoCV(cv=cv, n_jobs=-1, max_iter=20000, tol=1e-4, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate
lasso = model.named_steps["lassocv"]
print("Lasso Regression")
print("Optimal Alpha:", lasso.alpha_)
print("R^2:", model.score(X_val, y_val))
print()

filename = 'models/lasso_linear.joblib'
dump(model, filename)


# Configure Cross-Validation
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

# Fit ElasticNet
model = make_pipeline(
    StandardScaler(),
    ElasticNetCV(cv=cv, n_jobs=-1, max_iter=20000, tol=1e-4, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate
en = model.named_steps["elasticnetcv"]
print("Elastic Net Regression")
print(f"Best Alpha: {en.alpha_}")
print(f"Best l1_ratio: {en.l1_ratio_}")
print(f"Best Score (R^2 from CV): {model.score(X_val, y_val)}")
print()

filename = 'models/elastic_net.joblib'
dump(model, filename)

# Ridge CV
alphas = np.logspace(-6, 6, 13)

model = make_pipeline(StandardScaler(), 
                      RidgeCV(alphas=alphas, cv=5)
                     )

model.fit(X_train, y_train)

best_alpha = model.named_steps['ridgecv'].alpha_
best_score = model.named_steps['ridgecv'].best_score_
print("Ridge Regression")
print(f"Optimal alpha: {best_alpha}")
print(f"Best CV R^2 Score: {best_score}")

test_score = model.score(X_val, y_val)
print(f"Test set R^2 score: {test_score}")
print()


filename = 'models/ridge.joblib'
dump(model, filename)

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('regression', LinearRegression())
])

n = X.shape[0]
param_grid = {
    'pca__n_components': np.arange(1, n + 1)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kf,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_n_components = grid_search.best_params_['pca__n_components']
best_model = grid_search.best_estimator_
best_mse = -grid_search.best_score_

print("PCA Linear Regression")
print(f"Best number of components: {best_n_components}")
print(f"Best cross-validation MSE: {best_mse}")

val_mse = mean_squared_error(y_val, best_model.predict(X_val))
print(f"Validation set MSE: {val_mse}")

filename = 'models/pca_linear.joblib'
dump(model, filename)




