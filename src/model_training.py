from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def initialize_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression()
    elif model_name == "Random Forest Classifier":
        return RandomForestClassifier()
    elif model_name == "Linear Regression":
        return LinearRegression()
    elif model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor()
    elif model_name == "Gradient Boosting Classifier":
        return GradientBoostingClassifier()
    elif model_name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor()
    elif model_name == "KMeans Clustering":
        return KMeans(n_clusters=3)  # Default value

def train_model(data, target_column, model_name):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(target_column, axis=1), data[target_column])

    model = initialize_model(model_name)

    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    if model_name in ["Linear Regression", "Decision Tree Regressor", "Gradient Boosting Regressor"]:
        mae = mean_absolute_error(y_test, model.predict(X_test))
        mse = mean_squared_error(y_test, model.predict(X_test))
        r2 = r2_score(y_test, model.predict(X_test))
        return f"MAE: {mae}\nMSE: {mse}\nR2 Score: {r2}"
    elif model_name in ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"]:
        accuracy = accuracy_score(y_test, model.predict(X_test))
        precision = precision_score(y_test, model.predict(X_test))
        recall = recall_score(y_test, model.predict(X_test))
        f1 = f1_score(y_test, model.predict(X_test))
        return f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}"
    elif model_name == "KMeans Clustering":
        return f"Clusters: {model.labels_}"
