#importing packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

#Reading Dataset & Droping Useless Columns
df=pd.read_csv(r'py\NLP\House-Prices-Prediction\data\house_data.csv')
df=df.drop(columns=['date','id','zipcode'],axis=1)
df_sampled = df.sample(frac=0.1, random_state=42)  # Use 10% of the data

#defining our Features and target
X=df.drop(columns=['price'],axis=1)
y=df['price']

#Split Train and Test Data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#calling The model
# lr=LogisticRegression(random_state=42)
RFR=RandomForestRegressor(random_state=42)

#fiting the training data

RFR.fit(X_train,y_train)

#make prediction on our test data

pred=RFR.predict(X_test)

#evaluate the model on his own data
print(f"accuracy is :{r2_score(y_test,pred)}")
print(f"MSE is : {mean_squared_error(y_test,pred)}")