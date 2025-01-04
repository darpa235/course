
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def prepare_data():
    df = pd.read_csv('realty_data.csv')
    df= df.fillna(0)
    return df


def train_model(df):
    X, y = df[['total_square','rooms']], df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=500)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'flat_pr.pkl')
    return model

def predict_price(total_square,rooms):
    model = joblib.load('flat_pr.pkl')
    return model.predict([[total_square, rooms]])[0]

