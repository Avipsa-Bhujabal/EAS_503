```python
import csv
import sqlite3

# Function to load data from the CSV file
def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)
    return headers, data

# Load the data from the CSV file
headers, data = load_data('heart-2.csv')

# Connect to the database
conn = sqlite3.connect('heart_disease.db')
cursor = conn.cursor()

# Create and populate the heart_disease table
cursor.execute('''CREATE TABLE IF NOT EXISTS heart_disease (
    age INTEGER,
    sex TEXT,
    chest_pain INTEGER,
    resting_bp INTEGER,
    cholesterol INTEGER,
    fasting_blood_sugar BOOLEAN,
    resting_ecg INTEGER,
    max_heart_rate INTEGER,
    exercise_induced_chestpain TEXT,
    depress_w_ex_rest REAL,
    peak_ex_ST_seg INTEGER,
    coloured_vessels INTEGER,
    thalassemia INTEGER,
    presence_of_heartdisease INTEGER
)''')

# Insert data into the heart_disease table
for row in data:
    # Convert 'TRUE' and 'FALSE' to 1 and 0
    row[5] = 1 if row[5] == 'TRUE' else 0
    cursor.execute('''INSERT INTO heart_disease VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row)

# Create normalized tables in 3NF

# Patients table
cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex TEXT
)''')

# Heart conditions table
cursor.execute('''CREATE TABLE IF NOT EXISTS heart_conditions (
    condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chest_pain INTEGER,
    fasting_blood_sugar BOOLEAN,
    resting_ecg INTEGER,
    exercise_induced_chestpain TEXT,
    thalassemia INTEGER
)''')

# Test results table
cursor.execute('''CREATE TABLE IF NOT EXISTS test_results (
    test_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    condition_id INTEGER,
    resting_bp INTEGER,
    cholesterol INTEGER,
    max_heart_rate INTEGER,
    depress_w_ex_rest REAL,
    peak_ex_ST_seg INTEGER,
    coloured_vessels INTEGER,
    presence_of_heartdisease INTEGER,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (condition_id) REFERENCES heart_conditions(condition_id)
)''')

# Insert data into normalized tables
cursor.execute('''INSERT INTO patients (age, sex)
SELECT DISTINCT age, sex FROM heart_disease''')

cursor.execute('''INSERT INTO heart_conditions (chest_pain, fasting_blood_sugar, resting_ecg, exercise_induced_chestpain, thalassemia)
SELECT DISTINCT chest_pain, fasting_blood_sugar, resting_ecg, exercise_induced_chestpain, thalassemia FROM heart_disease''')

cursor.execute('''INSERT INTO test_results (patient_id, condition_id, resting_bp, cholesterol, max_heart_rate, depress_w_ex_rest, peak_ex_ST_seg, coloured_vessels, presence_of_heartdisease)
SELECT p.patient_id, hc.condition_id, hd.resting_bp, hd.cholesterol, hd.max_heart_rate, hd.depress_w_ex_rest, hd.peak_ex_ST_seg, hd.coloured_vessels, hd.presence_of_heartdisease
FROM heart_disease hd
JOIN patients p ON hd.age = p.age AND hd.sex = p.sex
JOIN heart_conditions hc ON hd.chest_pain = hc.chest_pain AND hd.fasting_blood_sugar = hc.fasting_blood_sugar AND hd.resting_ecg = hc.resting_ecg AND hd.exercise_induced_chestpain = hc.exercise_induced_chestpain AND hd.thalassemia = hc.thalassemia''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database created and data inserted successfully.")

```

    Database created and data inserted successfully.



```python
import sqlite3

# Connect to the database
conn = sqlite3.connect('heart_disease.db')
cursor = conn.cursor()

# Query the patients table
print("Sample data from patients table:")
cursor.execute("SELECT * FROM patients LIMIT 5")
print(cursor.fetchall())

# Query the heart_conditions table
print("\nSample data from heart_conditions table:")
cursor.execute("SELECT * FROM heart_conditions LIMIT 5")
print(cursor.fetchall())

# Query the test_results table
print("\nSample data from test_results table:")
cursor.execute("SELECT * FROM test_results LIMIT 5")
print(cursor.fetchall())

# Close the connection
conn.commit()
conn.close()

```

    Sample data from patients table:
    [(1, 52, 'Male'), (2, 53, 'Male'), (3, 70, 'Male'), (4, 61, 'Male'), (5, 62, 'Female')]
    
    Sample data from heart_conditions table:
    [(1, 0, 0, 1, 'no', 3), (2, 0, 1, 0, 'yes', 3), (3, 0, 0, 1, 'yes', 3), (4, 0, 1, 1, 'no', 2), (5, 0, 0, 0, 'no', 2)]
    
    Sample data from test_results table:
    [(1, 1, 1, 125, 212, 168, 1.0, 2, 2, 0), (2, 2, 2, 140, 203, 155, 3.1, 0, 0, 0), (3, 3, 3, 145, 174, 125, 2.6, 0, 0, 0), (4, 4, 1, 148, 203, 161, 0.0, 2, 1, 0), (5, 5, 4, 138, 294, 106, 1.9, 1, 3, 0)]



```python
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('heart_disease.db')

# SQL query to fetch data
sql_query = '''
SELECT 
    p.age, 
    p.sex, 
    hc.chest_pain, 
    tr.resting_bp, 
    tr.cholesterol, 
    hc.fasting_blood_sugar, 
    hc.resting_ecg, 
    tr.max_heart_rate, 
    hc.exercise_induced_chestpain, 
    tr.depress_w_ex_rest, 
    tr.peak_ex_ST_seg, 
    tr.coloured_vessels, 
    hc.thalassemia, 
    tr.presence_of_heartdisease
FROM 
    test_results tr
JOIN 
    patients p ON tr.patient_id = p.patient_id
JOIN 
    heart_conditions hc ON tr.condition_id = hc.condition_id
'''

# Load the data into a Pandas DataFrame
heart_disease_df = pd.read_sql_query(sql_query, conn)
conn.commit()
# Close the connection
conn.close()

# Display the first few rows of the DataFrame
print(heart_disease_df.head())

```

       age     sex  chest_pain  resting_bp  cholesterol  fasting_blood_sugar  \
    0   52    Male           0         125          212                    0   
    1   53    Male           0         140          203                    1   
    2   70    Male           0         145          174                    0   
    3   61    Male           0         148          203                    0   
    4   62  Female           0         138          294                    1   
    
       resting_ecg  max_heart_rate exercise_induced_chestpain  depress_w_ex_rest  \
    0            1             168                         no                1.0   
    1            0             155                        yes                3.1   
    2            1             125                        yes                2.6   
    3            1             161                         no                0.0   
    4            1             106                         no                1.9   
    
       peak_ex_ST_seg  coloured_vessels  thalassemia  presence_of_heartdisease  
    0               2                 2            3                         0  
    1               0                 0            3                         0  
    2               0                 0            3                         0  
    3               2                 1            3                         0  
    4               1                 3            2                         0  



```python
print(heart_disease_df)

```

          age     sex  chest_pain  resting_bp  cholesterol  fasting_blood_sugar  \
    0      52    Male           0         125          212                    0   
    1      53    Male           0         140          203                    1   
    2      70    Male           0         145          174                    0   
    3      61    Male           0         148          203                    0   
    4      62  Female           0         138          294                    1   
    ...   ...     ...         ...         ...          ...                  ...   
    1020   59    Male           1         140          221                    0   
    1021   60    Male           0         125          258                    0   
    1022   47    Male           0         110          275                    0   
    1023   50  Female           0         110          254                    0   
    1024   54    Male           0         120          188                    0   
    
          resting_ecg  max_heart_rate exercise_induced_chestpain  \
    0               1             168                         no   
    1               0             155                        yes   
    2               1             125                        yes   
    3               1             161                         no   
    4               1             106                         no   
    ...           ...             ...                        ...   
    1020            1             164                        yes   
    1021            0             141                        yes   
    1022            0             118                        yes   
    1023            0             159                         no   
    1024            1             113                         no   
    
          depress_w_ex_rest  peak_ex_ST_seg  coloured_vessels  thalassemia  \
    0                   1.0               2                 2            3   
    1                   3.1               0                 0            3   
    2                   2.6               0                 0            3   
    3                   0.0               2                 1            3   
    4                   1.9               1                 3            2   
    ...                 ...             ...               ...          ...   
    1020                0.0               2                 0            2   
    1021                2.8               1                 1            3   
    1022                1.0               1                 1            2   
    1023                0.0               2                 0            2   
    1024                1.4               1                 1            3   
    
          presence_of_heartdisease  
    0                            0  
    1                            0  
    2                            0  
    3                            0  
    4                            0  
    ...                        ...  
    1020                         1  
    1021                         0  
    1022                         0  
    1023                         1  
    1024                         0  
    
    [1025 rows x 14 columns]



```python
import pandas as pd
from ydata_profiling import ProfileReport

# Generate the profiling report

profile = ProfileReport(heart_disease_df, title="Heart Disease Dataset Profiling Report", explorative=True)

# Save the report as an HTML file
profile.to_file("heart_disease_profiling_report.html")

```


    Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]



    Generate report structure:   0%|          | 0/1 [00:00<?, ?it/s]



    Render HTML:   0%|          | 0/1 [00:00<?, ?it/s]



    Export report to file:   0%|          | 0/1 [00:00<?, ?it/s]



```python
%matplotlib inline
```


```python
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming `heart_disease_df` is already loaded into the environment

# Visualize the distribution of the target variable
sns.countplot(data=heart_disease_df, x='presence_of_heartdisease')
plt.title('Distribution of presence_of_heartdisease')
plt.show()

# Perform a stratified train/test split
train_df, test_df = train_test_split(
    heart_disease_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=heart_disease_df['presence_of_heartdisease']
)

# Output the proportion of each class in the train and test sets
train_counts = train_df['presence_of_heartdisease'].value_counts(normalize=True)
test_counts = test_df['presence_of_heartdisease'].value_counts(normalize=True)

print("Class proportions in training set:")
print(train_counts)
print("\nClass proportions in testing set:")
print(test_counts)

```


    
![png](output_6_0.png)
    


    Class proportions in training set:
    presence_of_heartdisease
    1    0.513415
    0    0.486585
    Name: proportion, dtype: float64
    
    Class proportions in testing set:
    presence_of_heartdisease
    1    0.512195
    0    0.487805
    Name: proportion, dtype: float64



```python
# Encoding 'sex' and 'exercise_induced_chestpain' to 1 and 0
heart_disease_df['sex'] = heart_disease_df['sex'].map({'Male': 0, 'Female': 1})
heart_disease_df['exercise_induced_chestpain'] = heart_disease_df['exercise_induced_chestpain'].map({'yes': 1, 'no': 0})

# Checking the changes
print(heart_disease_df[['sex', 'exercise_induced_chestpain']].head())

```

       sex  exercise_induced_chestpain
    0    0                           0
    1    0                           1
    2    0                           1
    3    0                           0
    4    1                           0



```python
# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming `heart_disease_df` is loaded in the environment

# Descriptive statistics for the dataset to explore distributions, capped values, and missing values
data_profile = heart_disease_df.describe(include='all').transpose()

# Checking for missing values
missing_values = heart_disease_df.isnull().sum()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = heart_disease_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Outputting descriptive statistics and missing values
data_profile, missing_values

```


    
![png](output_8_0.png)
    





    (                             count        mean        std    min    25%  \
     age                         1025.0   54.434146   9.072290   29.0   48.0   
     sex                         1025.0    0.304390   0.460373    0.0    0.0   
     chest_pain                  1025.0    0.942439   1.029641    0.0    0.0   
     resting_bp                  1025.0  131.611707  17.516718   94.0  120.0   
     cholesterol                 1025.0  246.000000  51.592510  126.0  211.0   
     fasting_blood_sugar         1025.0    0.149268   0.356527    0.0    0.0   
     resting_ecg                 1025.0    0.529756   0.527878    0.0    0.0   
     max_heart_rate              1025.0  149.114146  23.005724   71.0  132.0   
     exercise_induced_chestpain  1025.0    0.336585   0.472772    0.0    0.0   
     depress_w_ex_rest           1025.0    1.071512   1.175053    0.0    0.0   
     peak_ex_ST_seg              1025.0    1.385366   0.617755    0.0    1.0   
     coloured_vessels            1025.0    0.754146   1.030798    0.0    0.0   
     thalassemia                 1025.0    2.323902   0.620660    0.0    2.0   
     presence_of_heartdisease    1025.0    0.513171   0.500070    0.0    0.0   
     
                                   50%    75%    max  
     age                          56.0   61.0   77.0  
     sex                           0.0    1.0    1.0  
     chest_pain                    1.0    2.0    3.0  
     resting_bp                  130.0  140.0  200.0  
     cholesterol                 240.0  275.0  564.0  
     fasting_blood_sugar           0.0    0.0    1.0  
     resting_ecg                   1.0    1.0    2.0  
     max_heart_rate              152.0  166.0  202.0  
     exercise_induced_chestpain    0.0    1.0    1.0  
     depress_w_ex_rest             0.8    1.8    6.2  
     peak_ex_ST_seg                1.0    2.0    2.0  
     coloured_vessels              0.0    1.0    4.0  
     thalassemia                   2.0    3.0    3.0  
     presence_of_heartdisease      1.0    1.0    1.0  ,
     age                           0
     sex                           0
     chest_pain                    0
     resting_bp                    0
     cholesterol                   0
     fasting_blood_sugar           0
     resting_ecg                   0
     max_heart_rate                0
     exercise_induced_chestpain    0
     depress_w_ex_rest             0
     peak_ex_ST_seg                0
     coloured_vessels              0
     thalassemia                   0
     presence_of_heartdisease      0
     dtype: int64)




```python
# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Outlier Detection and Treatment (Example with cholesterol and resting_bp)
def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df_filtered

# Removing outliers from 'cholesterol' and 'resting_bp'
heart_disease_df = remove_outliers(heart_disease_df, 'cholesterol')
heart_disease_df = remove_outliers(heart_disease_df, 'resting_bp')

# 2. Categorical Encoding (Example for 'sex' and 'exercise_induced_chestpain')
encoder = LabelEncoder()
heart_disease_df['sex'] = encoder.fit_transform(heart_disease_df['sex'])
heart_disease_df['exercise_induced_chestpain'] = encoder.fit_transform(heart_disease_df['exercise_induced_chestpain'])

# 3. Feature Scaling (StandardScaler for continuous variables)
scaler = StandardScaler()
continuous_columns = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate']
heart_disease_df[continuous_columns] = scaler.fit_transform(heart_disease_df[continuous_columns])

# 4. Train/Test Split
X = heart_disease_df.drop('presence_of_heartdisease', axis=1)
y = heart_disease_df['presence_of_heartdisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Output the first few rows of the cleaned data
heart_disease_df.head()

# Plotting to visualize the effect of removing outliers
sns.boxplot(data=heart_disease_df, x='cholesterol')
plt.title("Cholesterol After Outlier Removal")
plt.show()

sns.boxplot(data=heart_disease_df, x='resting_bp')
plt.title("Resting BP After Outlier Removal")
plt.show()

```


    
![png](output_9_0.png)
    



    
![png](output_9_1.png)
    



```python
import mlflow
import dagshub
# Set up MLflow
mlflow.set_tracking_uri("https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow")
dagshub.init(repo_owner="Avipsa-Bhujabal", repo_name="my-first-repo")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Initialized MLflow to track repo <span style="color: #008000; text-decoration-color: #008000">"Avipsa-Bhujabal/my-first-repo"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Repository Avipsa-Bhujabal/my-first-repo initialized!
</pre>




```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn

# Define custom log transformation function
def log_transform(x):
    return np.log1p(x)

# Preprocessing pipeline
numeric_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate']
categorical_features = ['sex', 'chest_pain', 'exercise_induced_chestpain', 'resting_ecg', 'thalassemia']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('minmax', MinMaxScaler()),
    ('log_transform', FunctionTransformer(log_transform))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline with Logistic Regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train/test split
X = heart_disease_df.drop('presence_of_heartdisease', axis=1)
y = heart_disease_df['presence_of_heartdisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameters for tuning
param_grid = {'classifier__C': [0.01, 0.1, 1.0, 10.0], 'classifier__penalty': ['l2']}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='f1', n_jobs=-1)

# Set up MLflow experiment
mlflow.set_experiment("Heart Disease Prediction")

# Perform hyperparameter tuning and log results to MLflow
with mlflow.start_run(run_name="Experiment1"):
    # Log the dataset shape
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    
    # Perform the grid search
    grid_search.fit(X_train, y_train)
    
    # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log cross-validation results
    mlflow.log_metric('best_cv_score', grid_search.best_score_)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Log test set metrics
    mlflow.log_metric('test_f1_score', f1)
    mlflow.log_metric('test_accuracy', accuracy)
    mlflow.log_metric('true_positive', tp)
    mlflow.log_metric('true_negative', tn)
    mlflow.log_metric('false_positive', fp)
    mlflow.log_metric('false_negative', fn)
    
    # Log the best model
    mlflow.sklearn.log_model(best_model, "best_model")

# Output results summary
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{cm}")

```

    2024/12/21 13:32:28 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment1 at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/1b2a0e070bae42a4a26c1355510d580f
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    Best Hyperparameters: {'classifier__C': 10.0, 'classifier__penalty': 'l2'}
    Best CV F1-Score: 0.8184
    Test F1-Score: 0.8545
    Test Accuracy: 0.8418
    Confusion Matrix:
    [[74 20]
     [11 91]]



```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Train/test split
X = heart_disease_df.drop('presence_of_heartdisease', axis=1)
y = heart_disease_df['presence_of_heartdisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define classifiers (without XGBClassifier)
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RidgeClassifier': RidgeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
}

# Function to create and evaluate pipeline
def evaluate_classifier(clf_name, classifier):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    
    # Train on full training set
    pipeline.fit(X_train, y_train)
    
    # Predictions on test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Log results to MLflow
    with mlflow.start_run(run_name=f"Experiment_2_{clf_name}"):
        mlflow.log_param('classifier', clf_name)
        mlflow.log_metric('mean_cv_f1', np.mean(cv_scores))
        mlflow.log_metric('std_cv_f1', np.std(cv_scores))
        mlflow.log_metric('test_f1_score', f1)
        mlflow.log_metric('test_accuracy', accuracy)
        mlflow.log_metric('true_positive', tp)
        mlflow.log_metric('true_negative', tn)
        mlflow.log_metric('false_positive', fp)
        mlflow.log_metric('false_negative', fn)
        mlflow.log_param('confusion_matrix', cm.tolist())
        mlflow.sklearn.log_model(pipeline, "model")
    
    return pipeline, cv_scores, f1, accuracy, cm

# Evaluate each classifier
results = {}
for clf_name, classifier in classifiers.items():
    print(f"Evaluating {clf_name}...")
    results[clf_name] = evaluate_classifier(clf_name, classifier)

# Print summary of results
for clf_name, (_, cv_scores, f1, accuracy, cm) in results.items():
    print(f"\nResults for {clf_name}:")
    print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")

```

    Evaluating LogisticRegression...


    2024/12/21 13:38:41 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_2_LogisticRegression at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/a10d6597d6bb43f580a99ef873e97d80
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    Evaluating RidgeClassifier...


    2024/12/21 13:38:59 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_2_RidgeClassifier at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/8c49ca7051794fa5a04803fe3968e450
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    Evaluating RandomForestClassifier...


    2024/12/21 13:39:19 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_2_RandomForestClassifier at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/29b1d62cbf0f40eb9f0b31d98fc941c8
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    
    Results for LogisticRegression:
    Mean CV F1-Score: 0.8165 (+/- 0.0363)
    Test F1-Score: 0.8545
    Test Accuracy: 0.8418
    Confusion Matrix:
    [[74 20]
     [11 91]]
    
    Results for RidgeClassifier:
    Mean CV F1-Score: 0.8190 (+/- 0.0391)
    Test F1-Score: 0.8598
    Test Accuracy: 0.8469
    Confusion Matrix:
    [[74 20]
     [10 92]]
    
    Results for RandomForestClassifier:
    Mean CV F1-Score: 0.9864 (+/- 0.0046)
    Test F1-Score: 1.0000
    Test Accuracy: 1.0000
    Confusion Matrix:
    [[ 94   0]
     [  0 102]]



```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn

# Preprocessing pipeline
numeric_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate']
categorical_features = ['sex', 'chest_pain', 'exercise_induced_chestpain', 'resting_ecg', 'thalassemia']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Train/test split
X = heart_disease_df.drop('presence_of_heartdisease', axis=1)
y = heart_disease_df['presence_of_heartdisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply preprocessing manually
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb_model.fit(X_train_processed, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_processed)

# Calculate metrics
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Log results to MLflow
with mlflow.start_run(run_name="XGBoost_Experiment"):
    mlflow.log_param('classifier', 'XGBClassifier')
    mlflow.log_metric('test_f1_score', f1)
    mlflow.log_metric('test_accuracy', accuracy)
    mlflow.log_metric('true_positive', tp)
    mlflow.log_metric('true_negative', tn)
    mlflow.log_metric('false_positive', fp)
    mlflow.log_metric('false_negative', fn)
    mlflow.log_param('confusion_matrix', cm.tolist())
    mlflow.sklearn.log_model(xgb_model, "model")

# Print results
print(f"Test F1-Score: {f1:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{cm}")

```

    2024/12/21 13:46:49 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run XGBoost_Experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/f1db3faf1af445738639a3a7fe14185e
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    Test F1-Score: 1.0000
    Test Accuracy: 1.0000
    Confusion Matrix:
    [[ 94   0]
     [  0 102]]



```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np

# Feature engineering function
def engineer_features(X):
    X = X.copy()
    X['age_group'] = pd.cut(X['age'], bins=[0, 40, 60, 80, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
    X['bp_to_heartrate_ratio'] = X['resting_bp'] / X['max_heart_rate']
    X['cholesterol_level'] = pd.cut(X['cholesterol'], bins=[0, 200, 240, 1000], labels=['Normal', 'Borderline', 'High'])
    return X

# Define features
numeric_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'bp_to_heartrate_ratio']
categorical_features = ['sex', 'chest_pain', 'exercise_induced_chestpain', 'resting_ecg', 'thalassemia', 'age_group', 'cholesterol_level']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Feature engineering and preprocessing pipeline
feature_engineering = Pipeline([
    ('engineer', FunctionTransformer(engineer_features, validate=False)),
    ('preprocessor', preprocessor)
])
  

# Classifiers
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier()
}

# Assuming heart_disease_df is your dataset
X = heart_disease_df.drop('presence_of_heartdisease', axis=1)
y = heart_disease_df['presence_of_heartdisease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Evaluation function
def evaluate_model(clf_name, classifier, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('features', feature_engineering),
        ('classifier', classifier)
    ])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    with mlflow.start_run(run_name=f"Experiment_3_FeatureEng_{clf_name}"):
        mlflow.log_param('classifier', clf_name)
        mlflow.log_metric('mean_cv_f1', np.mean(cv_scores))
        mlflow.log_metric('std_cv_f1', np.std(cv_scores))
        mlflow.log_metric('test_f1_score', f1)
        mlflow.log_metric('test_accuracy', accuracy)
        mlflow.log_param('confusion_matrix', cm.tolist())
        
        if clf_name == 'RandomForestClassifier':
            feature_names = pipeline.named_steps['features'].named_steps['preprocessor'].get_feature_names_out()
            feature_imp = pd.DataFrame(pipeline.named_steps['classifier'].feature_importances_,
                                       index=feature_names,
                                       columns=['importance']).sort_values('importance', ascending=False)
            mlflow.log_param('top_features', feature_imp.head(10).to_dict())
        
        mlflow.sklearn.log_model(pipeline, "model")
    
    return pipeline, cv_scores, f1, accuracy, cm

# Evaluate each classifier
results = {}
for clf_name, classifier in classifiers.items():
    print(f"Evaluating {clf_name} with feature engineering...")
    results[clf_name] = evaluate_model(clf_name, classifier, X_train, X_test, y_train, y_test)

# Print summary of results
for clf_name, (_, cv_scores, f1, accuracy, cm) in results.items():
    print(f"\nResults for {clf_name} with feature engineering:")
    print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
```

    Evaluating LogisticRegression with feature engineering...


    2024/12/21 13:47:01 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_3_FeatureEng_LogisticRegression at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/63f139e77f274964b14fc97ab253df5e
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    Evaluating RandomForestClassifier with feature engineering...


    2024/12/21 13:47:18 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_3_FeatureEng_RandomForestClassifier at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/0d12c16a08cf43f4a323c22d69ec55a7
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    
    Results for LogisticRegression with feature engineering:
    Mean CV F1-Score: 0.8156 (+/- 0.0227)
    Test F1-Score: 0.8545
    Test Accuracy: 0.8418
    Confusion Matrix:
    [[74 20]
     [11 91]]
    
    Results for RandomForestClassifier with feature engineering:
    Mean CV F1-Score: 0.9827 (+/- 0.0071)
    Test F1-Score: 1.0000
    Test Accuracy: 1.0000
    Confusion Matrix:
    [[ 94   0]
     [  0 102]]



```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
# Custom Correlation Threshold Selector
class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Ensure X is a DataFrame for correlation computation
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.corr_matrix = X.corr().abs()
        return self

    def transform(self, X):
        # Ensure X is a DataFrame for dropping columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        upper_tri = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        return X.drop(columns=to_drop).values

# Feature selection pipeline
def create_feature_selection_pipeline(correlation_threshold, variance_threshold, k_best):
    return Pipeline([
        ('variance', VarianceThreshold(threshold=variance_threshold)),
        ('correlation', CorrelationThresholdSelector(threshold=correlation_threshold)),
        ('kbest', SelectKBest(f_classif, k=k_best))
    ])

# Preprocessing pipeline
numeric_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate']
categorical_features = ['sex', 'chest_pain', 'exercise_induced_chestpain', 'resting_ecg', 'thalassemia']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Classifiers
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier()
}

# Assuming heart_disease_df is your dataset
X = heart_disease_df.drop('presence_of_heartdisease', axis=1)
y = heart_disease_df['presence_of_heartdisease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Evaluation function
def evaluate_model(clf_name, classifier, X_train, X_test, y_train, y_test,
                   correlation_threshold, variance_threshold, k_best):
    feature_selection = create_feature_selection_pipeline(correlation_threshold,
                                                          variance_threshold,
                                                          k_best)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selection),
        ('classifier', classifier)
    ])
    
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        with mlflow.start_run(run_name=f"Experiment_4_FeatureSelection_{clf_name}"):
            mlflow.log_param('classifier', clf_name)
            mlflow.log_param('correlation_threshold', correlation_threshold)
            mlflow.log_param('variance_threshold', variance_threshold)
            mlflow.log_param('k_best', k_best)
            mlflow.log_metric('mean_cv_f1', np.mean(cv_scores))
            mlflow.log_metric('std_cv_f1', np.std(cv_scores))
            mlflow.log_metric('test_f1_score', f1)
            mlflow.log_metric('test_accuracy', accuracy)
            mlflow.log_param('confusion_matrix', cm.tolist())
        
        return pipeline, cv_scores, f1, accuracy, cm
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None, None, None

# Feature selection parameters
correlation_thresholds = [0.7]
variance_thresholds = [0.0]
k_best_values = [5]

# Evaluate each classifier with different feature selection parameters
results = {}
for clf_name, classifier in classifiers.items():
    for corr_thresh in correlation_thresholds:
        for var_thresh in variance_thresholds:
            for k in k_best_values:
                print(f"Evaluating {clf_name} with feature selection (Corr: {corr_thresh}, Var: {var_thresh}, K: {k})...")
                key = f"{clf_name}_Corr{corr_thresh}_Var{var_thresh}_K{k}"
                result = evaluate_model(clf_name,
                                        classifier,
                                        X_train,
                                        X_test,
                                        y_train,
                                        y_test,
                                        corr_thresh,
                                        var_thresh,
                                        k)
                if result[0] is not None:
                    results[key] = result

# Print summary of results
for key, (_, cv_scores, f1, accuracy, cm) in results.items():
    if cv_scores is not None:
        print(f"\nResults for {key}:")
        print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")

```

    Evaluating LogisticRegression with feature selection (Corr: 0.7, Var: 0.0, K: 5)...
    🏃 View run Experiment_4_FeatureSelection_LogisticRegression at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/646b84656d7645a180bca177a3841ce1
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    Evaluating RandomForestClassifier with feature selection (Corr: 0.7, Var: 0.0, K: 5)...
    🏃 View run Experiment_4_FeatureSelection_RandomForestClassifier at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11/runs/c7ce9c43845e4c1788110983bf5b5559
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/11
    
    Results for LogisticRegression_Corr0.7_Var0.0_K5:
    Mean CV F1-Score: 0.8163 (+/- 0.0236)
    Test F1-Score: 0.8341
    Test Accuracy: 0.8214
    Confusion Matrix:
    [[73 21]
     [14 88]]
    
    Results for RandomForestClassifier_Corr0.7_Var0.0_K5:
    Mean CV F1-Score: 0.8877 (+/- 0.0163)
    Test F1-Score: 0.9246
    Test Accuracy: 0.9235
    Confusion Matrix:
    [[89  5]
     [10 92]]



```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

numeric_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate']
categorical_features = ['sex', 'chest_pain', 'exercise_induced_chestpain', 'resting_ecg', 'thalassemia']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

def create_pca_pipeline(n_components):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components))
    ])

def plot_scree(pca):
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.grid()
    plt.show()

def evaluate_pca_model(clf_name, classifier, X_train, X_test, y_train, y_test, n_components):
    pca_pipeline = create_pca_pipeline(n_components)
    
    pipeline = Pipeline([
        ('pca_pipeline', pca_pipeline),
        ('classifier', classifier)
    ])
    
    pca_pipeline.fit(X_train)
    plot_scree(pca_pipeline.named_steps['pca'])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    with mlflow.start_run(run_name=f"Experiment_5_PCA_{clf_name}"):
        mlflow.log_param('classifier', clf_name)
        mlflow.log_param('n_components', n_components)
        mlflow.log_metric('mean_cv_f1', np.mean(cv_scores))
        mlflow.log_metric('std_cv_f1', np.std(cv_scores))
        mlflow.log_metric('test_f1_score', f1)
        mlflow.log_metric('test_accuracy', accuracy)
        mlflow.log_param('confusion_matrix', cm.tolist())
        
        explained_variance_ratio = pca_pipeline.named_steps['pca'].explained_variance_ratio_
        mlflow.log_param('explained_variance_ratio', explained_variance_ratio.tolist())
        
        mlflow.sklearn.log_model(pipeline, "model")
    
    print(f"Results for {clf_name} with {n_components} PCA components:")
    print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("\n")
    
    return pipeline, cv_scores, f1, accuracy, cm

# Prepare data
X = heart_disease_df.drop('presence_of_heartdisease', axis=1)
y = heart_disease_df['presence_of_heartdisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment
mlflow.set_experiment("Experiment_5_PCA")

# Define classifiers and PCA components
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier()
}
n_components_list = [2, 5]

# Evaluate models
for clf_name, classifier in classifiers.items():
    for n_components in n_components_list:
        evaluate_pca_model(clf_name, classifier, X_train, X_test, y_train, y_test, n_components)

```


    
![png](output_17_0.png)
    


    2024/12/21 13:48:13 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_5_PCA_LogisticRegression at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/5746ef4b6c824d7ea96a0eb91e767066
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for LogisticRegression with 2 PCA components:
    Mean CV F1-Score: 0.7506 (+/- 0.0465)
    Test F1-Score: 0.7668
    Test Accuracy: 0.7704
    Confusion Matrix:
    [[77 23]
     [22 74]]
    
    



    
![png](output_17_3.png)
    


    2024/12/21 13:48:30 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_5_PCA_LogisticRegression at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/86f0fe20709a49ee937e6b2e720f1a23
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for LogisticRegression with 5 PCA components:
    Mean CV F1-Score: 0.8003 (+/- 0.0391)
    Test F1-Score: 0.8205
    Test Accuracy: 0.8214
    Confusion Matrix:
    [[81 19]
     [16 80]]
    
    



    
![png](output_17_6.png)
    


    2024/12/21 13:48:49 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_5_PCA_RandomForestClassifier at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/5df686168a0e4e89963af3cad64cd730
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for RandomForestClassifier with 2 PCA components:
    Mean CV F1-Score: 0.9739 (+/- 0.0137)
    Test F1-Score: 1.0000
    Test Accuracy: 1.0000
    Confusion Matrix:
    [[100   0]
     [  0  96]]
    
    



    
![png](output_17_9.png)
    


    2024/12/21 13:49:06 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_5_PCA_RandomForestClassifier at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/1e594e6dbaad4704bbad9b68d6957825
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for RandomForestClassifier with 5 PCA components:
    Mean CV F1-Score: 0.9758 (+/- 0.0115)
    Test F1-Score: 1.0000
    Test Accuracy: 1.0000
    Confusion Matrix:
    [[100   0]
     [  0  96]]
    
    



```python
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

def evaluate_rfe_model(clf_name, classifier, X_train, X_test, y_train, y_test, n_features_to_select):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rfe', RFE(estimator=classifier, n_features_to_select=n_features_to_select)),
        ('classifier', classifier)
    ])
    
    try:
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        
        # Fit and predict on test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Metrics calculation
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Log results in MLflow
        with mlflow.start_run(run_name=f"RFE_{clf_name}_{n_features_to_select}_features"):
            mlflow.log_param('classifier', clf_name)
            mlflow.log_param('n_features_selected', n_features_to_select)
            mlflow.log_metric('mean_cv_f1', np.mean(cv_scores))
            mlflow.log_metric('std_cv_f1', np.std(cv_scores))
            mlflow.log_metric('test_f1_score', f1)
            mlflow.log_metric('test_accuracy', accuracy)
            mlflow.log_param('confusion_matrix', cm.tolist())
            
            # Log selected features if possible (only works if preprocessing doesn't change feature names)
            if hasattr(pipeline.named_steps['rfe'], 'support_'):
                selected_features_mask = pipeline.named_steps['rfe'].support_
                feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
                selected_features = [feature for feature, selected in zip(feature_names, selected_features_mask) if selected]
                mlflow.log_param('selected_features', selected_features)
            
            mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"Results for {clf_name} with {n_features_to_select} features:")
        print(f"Mean CV F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
    
    except Exception as e:
        print(f"Error occurred while evaluating {clf_name}: {e}")

# Number of features to select for RFE
n_features_list = [5, 10]

# Evaluate models with RFE for different numbers of selected features
for clf_name, classifier in classifiers.items():
    for n_features in n_features_list:
        evaluate_rfe_model(clf_name, classifier, X_train, X_test, y_train, y_test, n_features)

      
```

    2024/12/21 13:49:24 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run RFE_LogisticRegression_5_features at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/141acb36e9324f6d9ed4bcf869f43430
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for LogisticRegression with 5 features:
    Mean CV F1-Score: 0.8198 (+/- 0.0241)
    Test F1-Score: 0.8223
    Test Accuracy: 0.8214
    Confusion Matrix:
    [[80 20]
     [15 81]]


    2024/12/21 13:49:42 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run RFE_LogisticRegression_10_features at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/3e394cd3a43c449aae6ac3cc270f88d0
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for LogisticRegression with 10 features:
    Mean CV F1-Score: 0.8286 (+/- 0.0345)
    Test F1-Score: 0.8557
    Test Accuracy: 0.8520
    Confusion Matrix:
    [[81 19]
     [10 86]]


    2024/12/21 13:50:21 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run RFE_RandomForestClassifier_5_features at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/56ca1ecad92445e68eee17495b3fa159
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for RandomForestClassifier with 5 features:
    Mean CV F1-Score: 0.9794 (+/- 0.0152)
    Test F1-Score: 0.9841
    Test Accuracy: 0.9847
    Confusion Matrix:
    [[100   0]
     [  3  93]]


    2024/12/21 13:50:52 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run RFE_RandomForestClassifier_10_features at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/7679809c285245cba22f8ac1049abef1
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for RandomForestClassifier with 10 features:
    Mean CV F1-Score: 0.9805 (+/- 0.0105)
    Test F1-Score: 1.0000
    Test Accuracy: 1.0000
    Confusion Matrix:
    [[100   0]
     [  0  96]]



```python
from sklearn.model_selection import GridSearchCV


def evaluate_gridsearch_model(clf_name, classifier, param_grid):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_pipeline.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    with mlflow.start_run(run_name=f"Experiment_7_GridSearch_{clf_name}"):
        mlflow.log_param('classifier', clf_name)
        mlflow.log_params(best_params)
        mlflow.log_metric('best_cv_f1', grid_search.best_score_)
        mlflow.log_metric('test_f1_score', f1)
        mlflow.log_metric('test_accuracy', accuracy)
        mlflow.log_param('confusion_matrix', cm.tolist())
        
        mlflow.sklearn.log_model(best_pipeline, "model")
    
    print(f"Results for {clf_name} with GridSearchCV:")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("\n")
    
# Define hyperparameter grids
param_grids = {
    'LogisticRegression': {
        'classifier__C': [0.01, 0.1, 1.0],
        'classifier__penalty': ['l2']
    },
    'RandomForestClassifier': {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 10]
    }
}

# Evaluate models with GridSearchCV
for clf_name, param_grid in param_grids.items():
    evaluate_gridsearch_model(clf_name, classifiers[clf_name], param_grid)

```

    2024/12/21 13:51:04 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_7_GridSearch_LogisticRegression at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/e2c48f9dfded4f30a6123239f0598826
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for LogisticRegression with GridSearchCV:
    Best CV F1-Score: 0.8242
    Test F1-Score: 0.8557
    Test Accuracy: 0.8520
    Confusion Matrix:
    [[81 19]
     [10 86]]
    
    


    2024/12/21 13:51:21 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.


    🏃 View run Experiment_7_GridSearch_RandomForestClassifier at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7/runs/0260e0c3b77c4e82a62abba8f04a42b0
    🧪 View experiment at: https://dagshub.com/Avipsa-Bhujabal/my-first-repo.mlflow/#/experiments/7
    Results for RandomForestClassifier with GridSearchCV:
    Best CV F1-Score: 0.9770
    Test F1-Score: 1.0000
    Test Accuracy: 1.0000
    Confusion Matrix:
    [[100   0]
     [  0  96]]
    
    



```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow

# Fetch logged data from MLflow experiments
experiment_ids = [exp.experiment_id for exp in mlflow.search_experiments()]
results = []

for experiment_id in experiment_ids:
    runs = mlflow.search_runs([experiment_id])
    for _, run in runs.iterrows():
        results.append({
            'Experiment': run['tags.mlflow.runName'],
            'Mean CV F1-Score': run.get('metrics.mean_cv_f1', None),
            'Test F1-Score': run.get('metrics.test_f1_score', None)
        })

# Convert results to a DataFrame
df = pd.DataFrame(results)
df.dropna(subset=['Mean CV F1-Score', 'Test F1-Score'], inplace=True)

# Identify the best model
best_model = df.loc[df['Test F1-Score'].idxmax()]

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Experiment', y='Test F1-Score', data=df, palette='coolwarm', label='Test F1-Score', alpha=0.8)
sns.lineplot(x=range(len(df)), y='Mean CV F1-Score', data=df, color='darkblue', marker='o', label='Mean CV F1-Score', linewidth=2)

# Adding annotations
for index, row in df.iterrows():
    plt.text(index, row['Test F1-Score'] + 0.02, f"{row['Test F1-Score']:.2f}", ha='center', fontsize=8, color='black')
    plt.text(index, row['Mean CV F1-Score'] - 0.04, f"{row['Mean CV F1-Score']:.2f}", ha='center', fontsize=8, color='darkblue')

# Highlight the best model
plt.text(df.index[df['Test F1-Score'].idxmax()], best_model['Test F1-Score'] + 0.1, \
         f"Best: {best_model['Experiment']}\nF1: {best_model['Test F1-Score']:.2f}", \
         ha='center', fontsize=10, color='green', fontweight='bold')

plt.xticks(range(len(df)), [f"Exp {i+1}" for i in range(len(df))], rotation=45, ha='right', fontsize=8)
plt.title('Comparison of F1-Scores Across Experiments', fontsize=14, fontweight='bold')
plt.ylabel('F1-Score', fontsize=10)
plt.xlabel('Experiment', fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    



```python



```


```python

```
