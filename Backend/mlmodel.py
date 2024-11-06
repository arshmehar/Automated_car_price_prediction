# ### Importing libraries

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ### Loading Data

# %%
df = pd.read_csv("car_price_prediction.csv")
df


# ### Preprocessing


# #### Checking null values in dataset

# %%
df.isnull().sum()


# Checking total number of rows and columns in the dataset

# %%
df.shape

# %%
print(f"This DataSet Contains {df.shape[0]} rows & {df.shape[1]} columns")


# #### Checking datatype of the columns

# %%
df.info()


# #### Describing the dataset

# %%
df.describe(include = "all").T


# 1. The car price ranges from 1 t0 26307500 units.
# 2. The car production year starting from 1939 to 2020.
# 3. The most common car manufacturer in data is HYUNDAI.


# ## Data cleaning


# #### Droping the duplicate values

# %%
df1 = df.drop_duplicates(keep = False)


# #### Checking total number of rows and columns in the dataset after the droping of duplicate values

# %%
df1.shape

# %%
print(f"This DataSet Contains {df1.shape[0]} rows & {df1.shape[1]} columns after droping duplicate values")


# #### Checking the dataset after droping the duplicate values 

# %%
df1.head(10)


# It is observed that after droping the duplicate values some values contain '-' symbol instead of datatypes.

# %%
df1.info()


# #### Getting the columns which has dtypes objects

# %%
object_col = [col for col in df1.columns if df[col].dtype == 'object']
object_col


# #### Changing some columns dtype from object to integer and float

# %%
df1["Levy"] = df1["Levy"].replace("-","0")

# %%
df1["Levy"] = df1["Levy"].astype("int64")

# %%
df1["Levy"]

# %%
df1["Mileage"].head()

# %%
df1["Mileage"] = df1["Mileage"].str.replace(" km"," ")

# %%
df1["Mileage"] = df1["Mileage"].astype("int64")

# %%
df1["Mileage"]

# %%
df1["Doors"].unique()

# %%
df1["Doors"] = df1["Doors"].str.replace("04-May", "4-5")
df1["Doors"] = df1["Doors"].str.replace("02-Mar", "2-5")

# %%
df1["Doors"].unique()

# %%
df1["Engine volume"].unique()

# %%
df1["Engine volume"] = df1["Engine volume"].str.replace("Turbo", "")

# %%
df1["Engine volume"] = df1["Engine volume"].astype("float64")

# %%
df1["Engine volume"]

# %%
df1 = df1.drop("ID", axis = 1)


# Since Id doesn't have an any kind of information needed for the dataset so we drop the column ID.


# #### Checking of the Data after the cleaning 

# %%
df1


# • Here, Price column is the target variable in the dataset.
# 
# • And all other columns are the independent variable in the dataset.


# ## Visualization

# %%
for col in df1.select_dtypes(include = "object" ):
    print(f'The value count of the {col} is: \n\n {df1[col].value_counts()} \n\n')


# ### Analyzing fuel type used in the cars

# %%
df1["Fuel type"].unique()

# %%
fuelcounts = df1["Fuel type"].value_counts()
fuelcounts

# %%
sns.set_style("whitegrid")
plt.figure(figsize=(8,4))
sns.barplot(x = fuelcounts.index, y = fuelcounts, data = df1)
plt.xlabel("Fuel Type")
plt.ylabel("Number of cars")
plt.show()


# • The fuel type of petrol has maximum number of cars i.e 9803.
# 
# • And the fuel type of Hydrogen has least number of cars i.e 1.


# ### Analyzing that the car contains leather interior or not

# %%
df1["Leather interior"].unique()

# %%
Leather_interior = df1["Leather interior"].value_counts()
Leather_interior

# %%
sns.set_style('whitegrid')
sns.countplot(x = "Leather interior" , data = df1)
plt.title("No. of cars having Leather Interior")
plt.xlabel("Leather Interior")
plt.ylabel("No. of cars")
plt.show()

# %%
labels = ['Yes', 'No']
plt.pie(Leather_interior, autopct = "%1.2f%%", labels = labels)
plt.title("Leather interior or not")
plt.show()


# On the basis of the pie plot we can see that around 72.51% cars have leather interiors.


# ### Analyzing of different types of gear boxes used in the car

# %%
gearbox =  df1["Gear box type"].value_counts()
gearbox

# %%
plt.pie(gearbox, autopct = "%1.2f%%", labels = gearbox.index)
plt.title("Different types of Gearboxes")
plt.show()


# • We Analyze that 70.12% of Automatic type Gearboxes are installed in the car.
# 
# • We Analyze that 16.22% of Tiptronic type Gearboxes are installed in the car.
# 
# • We Analyze that 9.77% of Manual type Gearboxes are installed in the car.
# 
# • We Analyze that 3.88% of Variator type Gearboxes are installed in the car.

# %%
sns.barplot(x = gearbox.index,y = gearbox, data = df1)
plt.xlabel("Gear Box Type")
plt.ylabel("Number of Cars")
plt.show()


# • 13,116 vehicles have a Automatic gearbox.
# 
# • 3,034 vehicles have a Tiptronic gearbox.
# 
# • 1,828 vehicles have a Manual gearbox.
# 
# • 726 vehicles have a Variator gearbox.


# ### Analyzing Top 20 Car Manufacturers 

# %%
df1["Manufacturer"].value_counts()

# %%
manufacture = df1["Manufacturer"].value_counts().head(20)
manufacture

# %%
plt.figure(figsize=(12,6))
sns.set_style("darkgrid")
a = sns.barplot(x = manufacture.index,y = manufacture, data = df1);
a.set_xticklabels(manufacture.index ,rotation=90)
a.set(xlabel='Car Manufacturer', ylabel='Number of Cars')
plt.title("Number of cars Produced by the the top 20 Car Manufacturers")
plt.show()


# • Hyundai produced the maximum no. of cars and they are on the no. 1 position on producing the cars.
# 
# • Dodge produced the least no. of cars and they are on the no. 20 position on producing the cars.


# ### Analyzing the car on the basis of their Category i.e Body type

# %%
car_category = df1["Category"].value_counts()
car_category

# %%
plt.figure(figsize=(14,7))
sns.set_style("darkgrid")
a = sns.barplot(x = car_category.index,y = car_category, data = df1);
a.set_xticklabels(car_category.index ,rotation=90)
a.set(xlabel='Category/Body Type Name', ylabel='Number of Cars')
plt.title("Number of cars in each Category")
plt.show()


# • Sedan Category has maximum number of cars.
# 
# • Limousine Category has least number of cars.


# ### Analyzing  the Yearly Car Production

# %%
df1["Prod. year"].value_counts()

# %%
plt.figure(figsize=(15,12))
sns.countplot(x = "Prod. year", data = df1)
plt.xlabel("Production Year")
plt.ylabel("No. of Cars")
plt.title("No. of Cars Produced Yearly")
plt.xticks(rotation = 90)
plt.show()


# In Year 2012 maximum number of cars is produced. 


# ### Analysis of Engine Volume

# %%
df1["Engine volume"].value_counts()

# %%
plt.figure(figsize=(20,10))
sns.countplot(x = "Engine volume", data = df1)
plt.xlabel("Engine Volume")
plt.ylabel("No. of Cars")
plt.title("No. of Cars vs Engine Volume")
plt.xticks(rotation = 90)
plt.show()


# Majority of the car produced 2L Engine volume.


# ### Analyzing of the Mileage

# %%
df1["Mileage"].value_counts()

# %%
mileage = df1["Mileage"].value_counts().head(25)
mileage

# %%
plt.figure(figsize=(14,7))
a = sns.barplot(x = mileage.index,y = mileage, data = df1);
a.set_xticklabels(mileage.index ,rotation=90)
a.set(xlabel='Mileage', ylabel='Number of Cars')
plt.title("Number of cars Mileage(in km)")
plt.show()


# Most no. of cars are having the mileage 0 km.


#  ### Analysis on the Airbags

# %%
plt.figure(figsize=(14,7))
sns.histplot(x = df1["Airbags"])
plt.show()


# • Here count is no. of cars.
# 
# • It tells us that minimum 4 airbags are installed in the cars.

# %%
plt.figure(figsize=(12,6))
sns.boxplot(x = df1["Airbags"])
plt.show()



# ### Checking relationship of the columns

# %%
plt.figure(figsize=(16,14))
sns.heatmap(df1.corr(), annot = True, cmap = "Reds")
plt.show()


# ### Handeling with the outliers

# %%
for col in df1.select_dtypes(exclude = "object" ):
    plt.figure(figsize=(12,7))
    plt.subplot(1,2,1)
    sns.distplot(x = df1[col]);
    plt.title(f'The DistPlot for the {col} ')
    plt.subplot(1,2,2)
    sns.boxplot(x = df1[col]);
    plt.title(f'The BoxPlot for the {col} ')
    plt.suptitle(f'{col.title()} (Before handling outliers)',weight='bold')
    plt.show()


# Here, We can see that there are too many outliers in our dataset due to this we can't visualize the distplot & boxplot. These outliers are influencing the data distribution and making it challenging to draw meaningful conclusions from the visualizations.


# ### Checking number and percentage of the outliers present in our dataset.

# %%
def outlier_prcnt(data):
    for col_name in data.select_dtypes(exclude="object"):
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3 - q1  
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr

        outliers = ((data[col_name] > fence_high) | (data[col_name] < fence_low)).sum()
        total = data[col_name].shape[0]
        print(f"Total outliers in {col_name} are: {outliers} - {round(100 * (outliers) / total, 2)}%.")

outlier_prcnt(df1)

# %%
def outlier_handle(data):
    df1_copy = df1.copy()
    for col_name in data.select_dtypes(exclude="object"):
        q1 = data[col_name].quantile(0.25)
        q3 = data[col_name].quantile(0.75)
        iqr = q3-q1  #IQR
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df1_copy.loc[:,  col_name] = np.where(data[col_name]> fence_high, fence_high,
                                         np.where(data[col_name]< fence_low, fence_low,
                                                  data[col_name]))
    return df1_copy
df1 = outlier_handle(df1)

# %%
for col in df1.select_dtypes(exclude = "object" ):
    plt.figure(figsize=(12,7))
    plt.subplot(1,2,1)
    sns.distplot(x = df1[col]);
    plt.title(f'The DistPlot for the {col} ')
    plt.subplot(1,2,2)
    sns.boxplot(x = df1[col]);
    plt.title(f'The BoxPlot for the {col} ')
    plt.suptitle(f'{col.title()} (After handling outliers)',weight='bold')
    plt.show()

# %%
plt.figure(figsize=(15,15))
sns.pairplot(df1)
plt.show()


# ### Handeling outliers on skewness of Price columns

# %%
sns.distplot(df1['Price'], label = 'Skewness: %.2f'%(df1['Price'].skew()))
plt.legend(loc = 'best')
plt.title('Distribution of the Price column')
plt.show()


# ### Encoding

# %%
new_df = df1

# %%
from sklearn.preprocessing import LabelEncoder
Manufacturer_le = LabelEncoder()
Model_le = LabelEncoder()
Category_le = LabelEncoder()
Leather_interior_le = LabelEncoder()
Fuel_type_le = LabelEncoder()
Gear_box_type_le = LabelEncoder()
Doors_le  = LabelEncoder()
Wheel_le  = LabelEncoder() 
Drive_wheels_le = LabelEncoder() 
color_le = LabelEncoder()      

# %%
new_df['Manufacturer'] = Manufacturer_le.fit_transform(new_df['Manufacturer'])
new_df['Model'] = Model_le.fit_transform(new_df['Model'])
new_df['Category'] = Category_le.fit_transform(new_df['Category'])
new_df['Leather interior'] = Leather_interior_le.fit_transform(new_df['Leather interior'])
new_df['Fuel type'] = Fuel_type_le.fit_transform(new_df['Fuel type'])
new_df['Gear box type'] = Gear_box_type_le.fit_transform(new_df['Gear box type'])
new_df['Doors'] = Doors_le.fit_transform(new_df['Doors'])
new_df['Wheel'] = Wheel_le.fit_transform(new_df['Wheel'])
new_df['Drive wheels'] = Drive_wheels_le.fit_transform(new_df['Drive wheels'])

# %%
new_df = new_df.drop("Color", axis = 1)


# ## Data Modeling


# ### Linear Regression Model

# %%
new_df


# #### Splitting the Target and Data

# %%
X = new_df.iloc[:,1:]
Y = new_df['Price']

# %%
print(X)

# %%
print(Y)


# #### Splitting the Training and Test data

# %%
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y , test_size= 0.25 ,random_state=42)

# %%
print("x_train =", x_train.shape)
print("x_test =", x_test.shape)
print()
print("y_train =", y_train.shape)
print("y_test =", y_test.shape)


# Scaling the data

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# %%
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

# %%
y_pred = linear_regressor.predict(x_test)
lr = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
lr.head()


# #### Evaluation

# %%
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
mae = round(mean_absolute_error(y_test, y_pred), 3)
mse = round(mean_squared_error(y_test, y_pred), 3)
rmse = round(np.sqrt(mse), 3)

r2_value = round(r2_score(y_test, y_pred), 3)

print('Mean Absolute Error  of the model is : {}'.format(mae))
print('Mean Squared Error of the model is : {}'.format(mse))
print('Root Mean Squared Error of the model is : {}'.format(rmse))
print('R-squared value of the model is : {}'.format(r2_value))


# ### Logistic Regression Model

# %%
new_df 


# #### Splitting the Target and Data

# %%
X = new_df.iloc[:,1:-3]
Y = new_df["Wheel"]

# %%
X

# %%
Y


# #### Splitting the Training and Test data

# %%
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y , test_size= 0.30 ,random_state=25)

# %%
print("x_train =", x_train.shape)
print("x_test =", x_test.shape)
print()
print("y_train =", y_train.shape)
print("y_test =", y_test.shape)


# #### Scaling the data

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# %%
from sklearn.linear_model import LogisticRegression
logistic_regressor = LogisticRegression()
logistic_regressor.fit(x_train, y_train)

# %%
y_pred = logistic_regressor.predict(x_test)
lr = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
lr.head()

# %%
from sklearn.metrics import confusion_matrix

# %%
confusion_matrix(y_test,y_pred)

# %%
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

# %%
tn, fp, fn, tp

# %%
sns.heatmap(confusion_matrix(y_test,y_pred),fmt='d', annot=True)



# #### Evaluation


# ##### Accuracy

# %%
from sklearn.metrics import accuracy_score

# %%
accuracy_score(y_test,y_pred)


# ##### auc roc curve

# %%
from sklearn.metrics import roc_auc_score

# %%
roc_auc_score(y_test,y_pred)


# ##### classification report

# %%
from sklearn.metrics import classification_report

# %%
print(classification_report(y_test,y_pred))


# ### Decision Trees

# %%
new_df


# #### Splitting the Target and Data

# %%
X = new_df.drop(["Price", "Wheel"], axis = 1)
Y = new_df["Wheel"]

# %%
X

# %%
Y


# #### Splitting the Training and Test data

# %%
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y , test_size= 0.15 ,random_state=52)

# %%
print("x_train =", x_train.shape)
print("x_test =", x_test.shape)
print()
print("y_train =", y_train.shape)
print("y_test =", y_test.shape)

# %%
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

# %%
y_pred = dtc.predict(x_test)
lr = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
lr.head()

# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

# %%
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
tn, fp, fn, tp

# %%
sns.heatmap(confusion_matrix(y_test,y_pred), fmt='d', annot=True)
plt.show()


# ##### Accuracy

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# ##### Classification Report

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# Here total number of 1's is 2582 and 0's is 224.

# %%
plt.figure(figsize=(40,10))
from sklearn import tree
dtc = tree.DecisionTreeClassifier(max_depth=5)  
dtc.fit(x_train, y_train)
tree.plot_tree(dtc, feature_names = new_df.columns[:-2], class_names=["Left wheel","Right wheel"], filled=True)
plt.show()

# ### Random Forest Model

# %%
new_df

# #### Splitting the Target and Data

# %%
X = new_df.iloc[:,1:]
Y = new_df['Price']

# %%
X

# %%
Y

# #### Splitting the Training and Test data

# %%
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y , test_size= 0.25 ,random_state=42)

# %%
print("x_train =", x_train.shape)
print("x_test =", x_test.shape)
print()
print("y_train =", y_train.shape)
print("y_test =", y_test.shape)


# #### Scaling the Data

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# %%
from sklearn.ensemble import RandomForestRegressor
Rf = RandomForestRegressor(n_estimators = 400,max_depth=15, max_features='log2',random_state=1)
Rf.fit(x_train, y_train)

# %%
y_pred = Rf.predict(x_test)
y_pred = Rf.predict(x_test)
rf= pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
rf.head()


# %%
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
mae = round(mean_absolute_error(y_test, y_pred), 3)
mse = round(mean_squared_error(y_test, y_pred), 3)
rmse = round(np.sqrt(mse), 3)

r2_value = round(r2_score(y_test, y_pred), 3)

print('Mean Absolute Error  of the model is : {}'.format(mae))
print('Mean Squared Error of the model is : {}'.format(mse))
print('Root Mean Squared Error of the model is : {}'.format(rmse))
print('R-squared value of the model is : {}'.format(r2_value))
