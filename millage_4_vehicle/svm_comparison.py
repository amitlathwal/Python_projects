import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class DataPreprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def handle_missing_values(self, columns):
        self.imputer.fit(self.dataframe.iloc[:, columns])
        self.dataframe.iloc[:, columns] = self.imputer.transform(self.dataframe.iloc[:, columns])

    def encode_labels(self, columns):
        for column in columns:
            self.dataframe[column] = self.label_encoder.fit_transform(self.dataframe[column])

    def one_hot_encode(self, columns):
        column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns)],
                                               remainder='passthrough')
        self.dataframe = np.array(column_transformer.fit_transform(self.dataframe))

    def scale_features(self, columns):
        self.dataframe[:, columns] = self.scaler.fit_transform(self.dataframe[:, columns])

    def preprocess(self, impute_cols, label_encode_cols, one_hot_encode_cols, scale_cols):
        self.handle_missing_values(impute_cols)
        self.encode_labels(label_encode_cols)
        self.one_hot_encode(one_hot_encode_cols)
        self.scale_features(scale_cols)
        return self.dataframe


class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.results = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
            self.results[name] = rmse
        return self.results


def main():
    dataset = pd.read_csv('Four vehicle average.csv')
    X = dataset.iloc[:-1, 1:]
    y = dataset.iloc[:-1, 0]

    preprocessor = DataPreprocessor(X)
    X = preprocessor.preprocess(
        impute_cols=[0, 4, 7],
        label_encode_cols=['Engine_type', 'Transmission'],
        one_hot_encode_cols=[3, 5, 6],
        scale_cols=[8, 11, 12]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(kernel='poly'),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=10, random_state=1)
    }

    trainer = ModelTrainer(models)
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    print(results)


if __name__ == "__main__":
    main()
