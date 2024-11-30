#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



#import data
data=pd.read_csv('C:\\Users\\wwwza\\OneDrive\\Desktop\\project\\laptop_prices.csv')

#check info and datatype and print 5 rows
print(data.info())
print(data.head())

#check for null values
print(data.isnull().sum())

#univariate analysis
sns.barplot=data['Company'].value_counts().plot(kind = 'bar')
plt.title('Number of products')
plt.show() 


plt.figure(figsize = (10,8))
sns.boxplot(x = data['OS'], y= data['Price_euros'])
plt.title('OS VS CPU_freqency')
plt.show() 


#plot values by company and price
viz1=pd.DataFrame(data)
plt.figure(figsize=(10,5))
viz1=data.groupby('Company')['Price_euros'].mean().reset_index().sort_values('Price_euros', ascending=False)
sns.barplot(x='Price_euros',y='Company', data=viz1)
plt.title('Brand V/s Price')
plt.xlabel('Price')
plt.ylabel('Company')
plt.show()


data=pd.DataFrame(data)
plt.figure(figsize=(10,5))
sns.catplot(x='CPU_company', y='Price_euros', kind='bar', hue='OS', data=data, height=6, aspect=1.5)
plt.title('Price Distribution by Features and OS')
plt.xlabel('Features')
plt.ylabel('Price (Euros)')
plt.show()

#linear regression
from sklearn.model_selection import train_test_split
# Convert categorical columns to numerical using One-Hot Encoding
X = data.drop(columns=['Company'], axis=1)
y = data['Price_euros']
X=pd.get_dummies(X, drop_first=True)


data=pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

#visualize scatter plot for test and predicted data
plt.scatter(y_test,y_pred, color='green')
plt.xlabel('True Values (y_test)')
plt.ylabel('Predictions (y_pred)')
plt.title('Linear Regression: True vs Predicted Values')
plt.show()
