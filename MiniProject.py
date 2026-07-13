import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df
print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data

#process data
selected_features = ['fracSpent', 'fracComp', 'fracPlayed', 'fracPaused', 'numPauses', 'avgPBR', 'stdPBR', 'numRWs', 'numFFs', 's']
dataset = df[selected_features]

#handle the missing values
imputer = SimpleImputer(strategy='mean')
dataset_imputed = pandas.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

#scaling
scaler = StandardScaler()
dataset_scaled = pandas.DataFrame(scaler.fit_transform(dataset_imputed), columns=dataset_imputed.columns)

#question 1: clusters
print("Question 1: Cluster Data")

#kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=12)  
kmeans.fit(dataset_scaled)

#cluster labels
dataset_scaled['Cluster'] = kmeans.labels_

#print class data
label = {0: 'Incorrect', 1: 'Correct'}
dataset_label = pandas.DataFrame()
dataset_label['Class'] = dataset_scaled['Cluster'].map(label)
print(dataset_label.head(10))

#question 2: predictions on performance
print("Question 2: Predictions on Performance")

#split into x and y data
X_regression = dataset_scaled.drop(columns=['s', 'Cluster'])
y_regression = dataset_scaled['s']

#testing and training sets for evaluation
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.1, random_state=12)

#linear regression model
regression_model = LinearRegression()
regression_model.fit(X_train_regression, y_train_regression)

#predict testing data
y_pred_regression = regression_model.predict(X_test_regression)

#evaluate the regression model
mse_regression = mean_squared_error(y_test_regression, y_pred_regression)
print("Mean Squared Error:", mse_regression)

#question 3: predict performance based on a particular video
print("Question 3: Performance on a Particular Video")

###
import matplotlib.pyplot as plt

#plot data points
plt.scatter(df['numPauses'], df['fracPaused'], c=df['s'])
plt.xlabel('numPauses')
plt.ylabel('fracPaused')
plt.colorbar(label='s')
plt.show()

plt.scatter(df['fracComp'], df['fracPlayed'], c=df['s'])
plt.xlabel('fracComp')
plt.ylabel('fracPlayed')
plt.colorbar(label='s')
plt.show()

plt.scatter(df['avgPBR'], df['stdPBR'], c=df['s'])
plt.xlabel('avgPBR')
plt.ylabel('stdPBR')
plt.colorbar(label='s')
plt.show()

plt.scatter(df['numRWs'], df['numFFs'], c=df['s'])
plt.xlabel('numRWs')
plt.ylabel('numFFs')
plt.colorbar(label='s')
plt.show()

#split the data
X = dataset_scaled.drop('s', axis=1)
y = dataset_scaled['s']

#split the data, train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

#scale using imported fx
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_regr = KNeighborsRegressor(n_neighbors=3)
knn_regr.fit(X_train_scaled, y_train)

y_pred_regression = knn_regr.predict(X_test_scaled)
mse_regression = mean_squared_error(y_test, y_pred_regression)
print('MSE Regression:',mse_regression)

###KNeighbors
#one
X = dataset_scaled.drop(['numPauses','fracPaused'], axis=1)
y = df['s']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

k = 6
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report [numPauses - fracPaused]:")
print(classification_report(y_test, y_pred))

#two
X = dataset_scaled.drop(['fracComp','fracPlayed'], axis=1)
y = df['s']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

k = 6
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report [fracComp - fracPlayed]:")
print(classification_report(y_test, y_pred))

#three
X = dataset_scaled.drop(['avgPBR','stdPBR'], axis=1)
y = df['s']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

k = 6
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report [avgPBR - stdPBR]:")
print(classification_report(y_test, y_pred))

#fourth
X = dataset_scaled.drop(['numRWs','numFFs'], axis=1)
y = df['s']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

k = 6
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report [numRWs - numFFs]:")
print(classification_report(y_test, y_pred))