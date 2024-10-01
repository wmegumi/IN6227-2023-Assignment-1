
import pandas as pd
import math
from collections import defaultdict
import numpy as np
#load and preprocess data
def load_data(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'salary']

    data = pd.read_csv(path, header=None, names=column_names)
    data.dropna(inplace=True)
    # 编码分类变量
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                           'native-country', 'salary']

    for column in categorical_columns:
        unique_values = sorted(data[column].unique())

        mapping = {value: idx for idx, value in enumerate(unique_values)}
        data[column] = data[column].map(mapping)


    # standardize continuous features
    continuous_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for column in continuous_columns:
        mean = data[column].mean()
        std = data[column].std()
        data[column] = (data[column] - mean) / std

    # split data into X and y
    x = data.drop('salary', axis=1)
    y = data['salary']

    return x, y


#classifiers
# Naive Bayes Classifier
class NaiveBayes:
    def __init__(self,var_smoothing,laplace_smoothing,class_weights):
        self.priors = None
        self.var = None
        self.mean = None
        self.classes = None
        self.var_smoothing = var_smoothing
        self.laplace_smoothing = laplace_smoothing
        self.class_weights = class_weights

    def fit(self, X, y):
        self.classes = list(set(y))
        self.mean = defaultdict(list)
        self.var = defaultdict(list)
        self.priors = {}

        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i] == c]
            self.mean[c] = [sum(col) / len(col) for col in zip(*X_c)]
            self.var[c] = [sum((x - m) ** 2 for x in col) / len(col) for col, m in zip(zip(*X_c), self.mean[c])]+[self.var_smoothing]
            self.priors[c] = (len(X_c) + self.laplace_smoothing) / (len(X) + self.laplace_smoothing * len(self.classes))
            if self.class_weights:
                self.priors[c] *= self.class_weights.get(c, 1)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []
        epsilon = 1e-9  # Small constant to avoid log(0)
        for c in self.classes:
            prior = math.log(self.priors[c] + epsilon)
            posterior = sum(math.log(self._pdf(c, x[i], i) + epsilon) for i in range(len(x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[posteriors.index(max(posteriors))]

    def _pdf(self, class_idx, x, i):
        mean = self.mean[class_idx][i]
        var = self.var[class_idx][i]
        numerator = math.exp(- (x - mean) ** 2 / (2 * var))
        denominator = math.sqrt(2 * math.pi * var)
        return numerator / denominator


def downsample_data(X, y):
    # Combine X and y into a single DataFrame
    data = pd.concat([X, y], axis=1)

    # Separate majority and minority classes
    majority_class = data[data.salary == 0]
    minority_class = data[data.salary == 1]
    np.random.seed(42)
    majority_class_downsampled = majority_class.sample(n=len(minority_class), replace=False, random_state=42)
    downsampled = pd.concat([majority_class_downsampled, minority_class])

    # Separate X and y
    X_downsampled = downsampled.drop('salary', axis=1)
    y_downsampled = downsampled['salary']

    return X_downsampled, y_downsampled
def upsample_minority_class(X, y):
    # Combine X and y into a single DataFrame
    data = pd.concat([X, y], axis=1)

    # Separate majority and minority classes
    majority_class = data[data.salary == 0]
    minority_class = data[data.salary == 1]

    # Upsample minority class
    np.random.seed(42)
    minority_class_upsampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)

    # Combine majority class with upsampled minority class
    upsampled = pd.concat([majority_class, minority_class_upsampled])

    # Separate X and y
    X_upsampled = upsampled.drop('salary', axis=1)
    y_upsampled = upsampled['salary']

    return X_upsampled, y_upsampled
#evaluation
def classification_report(y_true, y_pred):

    classes = np.unique(y_true)
    report = {}

    for cls in classes:
        tp = sum((y_true == cls) & (y_pred == cls))
        fp = sum((y_true != cls) & (y_pred == cls))
        fn = sum((y_true == cls) & (y_pred != cls))
        tn = sum((y_true != cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': tp + fn
        }

    return report
if __name__ == "__main__":
    x_train,y_train=load_data('data/adult.data')
    x_train, y_train = upsample_minority_class(x_train, y_train)
    model = NaiveBayes(1,1,{0: 1, 1: 1})
    model.fit(x_train.values, y_train.values)
    x_test,y_test=load_data('data/adult.test')
    y_pred = model.predict(x_test.values)
    print(classification_report(y_test.values, y_pred))
    accuracy = np.mean(y_pred == y_test.values)
    print(f"Model accuracy: {accuracy:.2f}")

