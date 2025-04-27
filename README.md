# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## Developed By: MAHALAKSHMI R
## Register no: 212223230116

# CODING AND OUTPUT:
```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![1](https://github.com/user-attachments/assets/d55b4e52-fa30-424e-a709-a884301363d8)

```py
df_null_sum=df.isnull().sum()
df_null_sum
```
![2](https://github.com/user-attachments/assets/835e2655-7fbf-4d05-9f98-1d661b8c848b)

```py
df.dropna()
```
![3](https://github.com/user-attachments/assets/e8f74ea8-a438-4e1a-b2ff-8dedb76eca95)

```py
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![4](https://github.com/user-attachments/assets/fbb874cc-6739-44a7-9804-4cad93a23369)

```py
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![5](https://github.com/user-attachments/assets/c27d093a-3cc4-41a2-9341-2afd047cf835)


```py
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![6](https://github.com/user-attachments/assets/e4dea2fd-b2a2-4a4c-80ae-bcbf50823fce)

```py
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![7](https://github.com/user-attachments/assets/ea366337-e676-4583-9834-afb9636c7b06)

```py
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```
![8](https://github.com/user-attachments/assets/e4951843-4dec-445e-a870-6c1e96d7122c)

```py
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![9](https://github.com/user-attachments/assets/89ca4ab8-06d7-4dc2-9809-4a720f0e18d3)

```py
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```
![10](https://github.com/user-attachments/assets/9fbd860a-1b80-4fca-afeb-211bc140dc9c)

```py
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![11](https://github.com/user-attachments/assets/a4888db8-3689-4700-b9d1-15e3639b26ac)

```py
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![12](https://github.com/user-attachments/assets/0822cde6-1c19-4184-8ca8-85c4f2a27105)

```py
df
```
![13](https://github.com/user-attachments/assets/8997cb5a-1c18-4895-954a-18b1f92cb174)

```py
df.info()
```
![14](https://github.com/user-attachments/assets/ff426ef5-b67d-48ba-8717-9dd9e64f4ef2)

```py
df_null_sum=df.isnull().sum()
df_null_sum
```

![15](https://github.com/user-attachments/assets/c421d572-7b3e-4964-b89f-c15678c7cb15)

```py
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![16](https://github.com/user-attachments/assets/ecdcdeae-3400-4e75-9952-80c10e5b814b)

```py
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![17](https://github.com/user-attachments/assets/a2ee9497-39ed-43f6-87e4-ef0684f05876)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

```py
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![18](https://github.com/user-attachments/assets/4c7b4aa7-8b06-4fb2-9f3c-cd5963ffdb61)

```py
y_pred = rf.predict(X_test)
```

```py
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![19](https://github.com/user-attachments/assets/be50b410-3fea-4e42-9e3c-4beef384b784)

```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![20](https://github.com/user-attachments/assets/47eaaf22-5f83-408d-aa3d-7a4ebbc8acf2)

```py
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![21](https://github.com/user-attachments/assets/e77656c7-f097-43ca-a79f-d302cd06ffc2)


```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```py
k_chi2 = 6
```

```py
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
```

```py
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![22](https://github.com/user-attachments/assets/3430678b-2ed3-416d-8eb7-47b3c4e5c87e)

```py
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

```py
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![23](https://github.com/user-attachments/assets/89af7813-b60a-4a5a-8df6-7c282537a294)

```py
y_pred = rf.predict(X_test)
```

```py
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![24](https://github.com/user-attachments/assets/a38a9599-fc8a-4b80-9808-47a69e3396ef)

```py
!pip install skfeature-chappers
```
![25](https://github.com/user-attachments/assets/c0570926-c136-47d4-ae79-dd1cda92cfe8)

```py
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

```py
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![26](https://github.com/user-attachments/assets/dfba0ec7-286e-4ff2-b979-8e66bb3fb9b1)

```py
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![27](https://github.com/user-attachments/assets/053d47e9-0d6e-4d98-b5a1-2051a422df8b)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```py
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```

```py
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![28](https://github.com/user-attachments/assets/25c44f04-1713-44c6-9912-369ebd88587d)

```py
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![29](https://github.com/user-attachments/assets/e36baeb6-f93f-46b3-b80e-44829484b644)

```py
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![30](https://github.com/user-attachments/assets/672e508c-af51-43bc-94f6-c0078eecade4)

```py
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```

```py
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![31](https://github.com/user-attachments/assets/d2229637-49c1-462d-a542-2d924985b001)


```py
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![32](https://github.com/user-attachments/assets/5080ea4f-eb4a-4d4d-8fe5-a3eefa98eeed)

```py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```

```py
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
```

```py
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

```py
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![33](https://github.com/user-attachments/assets/42dfaf4b-6c6d-4b9b-8f26-2495a32d801c)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
