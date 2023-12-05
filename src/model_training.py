from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

def train_model(data, target_column, model_name):
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(data.drop(target_column, axis=1), data[target_column])

    # Initialize and train model
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor()

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    return f"{model_name} Accuracy: {accuracy * 100:.2f}%"
