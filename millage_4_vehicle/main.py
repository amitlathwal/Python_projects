import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from category_encoders import BinaryEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import cross_val_score

car_df = pd.read_csv('Four vehicle average.csv')

for i in list(car_df.keys()):
    car_df[i] = car_df[i].replace('NA', np.nan)

columns = ['Engine_type', 'Transmission', 'Mostly_driven', 'Way_of_driving', 'Average_Load']

for i in columns:
    encoder = BinaryEncoder(cols=[i])
    newdata = encoder.fit_transform(car_df[i])
    car_df = pd.concat([car_df, newdata], axis=1)
    car_df = car_df.drop([i], axis=1)

x = car_df.iloc[:-1, 1:]
y = car_df.iloc[:-1, 0]
kf = KFold(n_splits=5, shuffle=True, random_state=2)

results = []
LR = LinearRegression()
RF = RandomForestRegressor()
KN = KNeighborsRegressor()
DT = DecisionTreeRegressor()
models = [LR, RF, KN, DT]

sample = [[15, 1200, 1100, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]]  # i20, petrol, year 2015, 1200cc, Manual, 1100Kg

for model in models:
    for train, test in kf.split(x, y):
        xtrain = x.iloc[train]
        xtest = x.iloc[test]
        ytrain = y.iloc[train]
        ytest = y.iloc[test]

        model.fit(xtrain, ytrain)
        ypredict = model.predict(xtest)
        results.append(np.sqrt(metrics.mean_squared_error(ytest, ypredict)))

    print(results)
    print(np.mean(results))
    print(np.var(results, ddof=1))
    print()

    results = []
    new_mileage = model.predict(sample)
    print("Sample's millage is " + str(new_mileage[0]))
    print()

'''
models=[]
models.append(('LR', LinearRegression()))
models.append(('RF', RandomForestRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))

for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    cv_results = cross_val_score(model, x, y, cv=kfold)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
'''
