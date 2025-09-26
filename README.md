# TASK-03
Housing Price Prediction using Linear Regression

This project demonstrates a simple Housing Price Prediction using Linear Regression in Python. It covers data preprocessing, exploratory data analysis (EDA), model building, and evaluation.


---

Table of Contents

Overview

Dataset

Libraries Used

Steps Implemented

1. Load Dataset

2. Preview Dataset

3. Data Preprocessing

4. Exploratory Data Analysis (EDA)

5. Model Building

6. Model Evaluation

7. Visualization


Usage

Results

License



---

Overview

This project builds a linear regression model to predict housing prices based on various features. It includes:

Handling missing values

Encoding categorical variables

Feature scaling

Exploratory Data Analysis (EDA)

Linear Regression model training and evaluation

Visualization of Actual vs Predicted prices



---

Dataset

The dataset used is a CSV file (Housing (13).csv) containing housing-related features. The target variable is assumed to be price.


---

Libraries Used

pandas – Data manipulation

numpy – Numerical operations

matplotlib – Data visualization

seaborn – Advanced visualizations

scikit-learn – Machine learning, preprocessing, and model evaluation



---

Steps Implemented

1. Load Dataset

The dataset is loaded using pandas.read_csv():

df = pd.read_csv("Housing (13).csv")

2. Preview Dataset

Check the first few rows, dataset info, missing values, and statistical summary:

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

3. Data Preprocessing

Encode categorical variables using LabelEncoder

Fill missing values with the mean

Standardize features using StandardScaler


categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = label_enc.fit_transform(df[col])

df.fillna(df.mean(), inplace=True)
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

4. Exploratory Data Analysis (EDA)

Correlation heatmap

Histograms of features


sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
df.hist(figsize=(12, 10), bins=30)

5. Model Building

Separate features (X) and target (y)

Split dataset into training and testing sets

Train a Linear Regression model


X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

6. Model Evaluation

Evaluate the model using Mean Squared Error (MSE) and R-squared Score:

y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

7. Visualization

Plot Actual vs Predicted prices
