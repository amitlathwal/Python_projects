import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from category_encoders import BinaryEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class Preprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def handle_missing_values(self):
        self.dataframe.replace('NA', np.nan, inplace=True)

    def encode_columns(self, columns):
        for column in columns:
            encoder = BinaryEncoder(cols=[column])
            encoded_data = encoder.fit_transform(self.dataframe[column])
            self.dataframe = pd.concat([self.dataframe, encoded_data], axis=1)
            self.dataframe.drop([column], axis=1, inplace=True)

    def get_features_and_target(self):
        x = self.dataframe.iloc[:-1, 1:]
        y = self.dataframe.iloc[:-1, 0]
        return x, y


class ModelTrainer:
    def __init__(self, models, sample):
        self.models = models
        self.sample = sample
        self.results = []

    def train_and_evaluate(self, x, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        for model in self.models:
            model_name = type(model).__name__
            print(f"Evaluating {model_name}")
            model_results = []
            for train_index, test_index in kf.split(x):
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                model_results.append(rmse)

            mean_rmse = np.mean(model_results)
            variance_rmse = np.var(model_results, ddof=1)
            self.results.append((model_name, mean_rmse, variance_rmse))

            print(f"Results for {model_name}:")
            print(f"Mean RMSE: {mean_rmse}")
            print(f"Variance RMSE: {variance_rmse}\n")

            sample_prediction = model.predict(self.sample)
            print(f"Sample's mileage prediction by {model_name}: {sample_prediction[0]}\n")

    def cross_validate(self, x, y):
        for model in self.models:
            kfold = KFold(n_splits=5, shuffle=True, random_state=2)
            cv_results = cross_val_score(model, x, y, cv=kfold)
            mean_score = cv_results.mean()
            std_score = cv_results.std()
            name = type(model).__name__
            print(f"{name}: {mean_score:.6f} ({std_score:.6f})")


def main():
    car_df = pd.read_csv('Four vehicle average.csv')

    preprocessor = Preprocessor(car_df)
    preprocessor.handle_missing_values()
    columns_to_encode = ['Engine_type', 'Transmission', 'Mostly_driven', 'Way_of_driving', 'Average_Load']
    preprocessor.encode_columns(columns_to_encode)
    x, y = preprocessor.get_features_and_target()

    sample = pd.DataFrame([[15, 1200, 1100, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]], columns=x.columns)  # i20, petrol, year 2015, 1200cc, Manual, 1100Kg
    models = [LinearRegression(), RandomForestRegressor(), KNeighborsRegressor(), DecisionTreeRegressor()]
    model_trainer = ModelTrainer(models, sample)
    model_trainer.train_and_evaluate(x, y)
    # model_trainer.cross_validate(x, y)


if __name__ == "__main__":
    main()
