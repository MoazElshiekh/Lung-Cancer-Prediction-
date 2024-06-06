# Lung-Cancer-Prediction-
# Lung Cancer Survey Dataset Analysis

## Introduction
This project involves analyzing a dataset from a lung cancer survey. The dataset contains various features related to the health and habits of individuals, which will be used to study correlations and possibly predict lung cancer risk.

## Step 1: Describe the Dataset and Its Features

### Loading and Describing the Dataset
The first step is to load the dataset and examine its structure and features. The dataset is in CSV format and is loaded using the pandas library in Python.

```python
import pandas as pd

# Load the dataset
SurveyLungCancer_df = pd.read_csv("SurveyLungCancer.csv")

# Describe the datasets and their features
print("\n1.SurveyLungCancer :")
print(SurveyLungCancer_df.info())
```

### Output Description
The output from the above code provides a summary of the dataset's structure:

```
1.SurveyLungCancer :
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 309 entries, 0 to 308
Data columns (total 16 columns):
 #   Column                 Non-Null Count  Dtype 
---  ------                 --------------  ----- 
 0   GENDER                 309 non-null    object
 1   AGE                    309 non-null    int64 
 2   SMOKING                309 non-null    int64 
 3   YELLOW_FINGERS         309 non-null    int64 
 4   ANXIETY                309 non-null    int64 
 5   PEER_PRESSURE          309 non-null    int64 
 6   CHRONIC DISEASE        309 non-null    int64 
 7   FATIGUE                309 non-null    int64 
 8   ALLERGY                309 non-null    int64 
 9   WHEEZING               309 non-null    int64 
 10  ALCOHOL CONSUMING      309 non-null    int64 
 11  COUGHING               309 non-null    int64 
 12  SHORTNESS OF BREATH    309 non-null    int64 
 13  SWALLOWING DIFFICULTY  309 non-null    int64 
 14  CHEST PAIN             309 non-null    int64 
 15  LUNG_CANCER            309 non-null    object
dtypes: int64(14), object(2)
memory usage: 38.8+ KB
None
```

### Dataset Information
- **Total Entries**: 309
- **Total Features**: 16
- **Feature Types**: 
  - 14 features are of type `int64`
  - 2 features are of type `object`

### Feature List
1. **GENDER**: Gender of the individual (object)
2. **AGE**: Age of the individual (int64)
3. **SMOKING**: Smoking habit (int64)
4. **YELLOW_FINGERS**: Presence of yellow fingers (int64)
5. **ANXIETY**: Anxiety level (int64)
6. **PEER_PRESSURE**: Influence of peer pressure (int64)
7. **CHRONIC DISEASE**: Presence of chronic disease (int64)
8. **FATIGUE**: Fatigue level (int64)
9. **ALLERGY**: Presence of allergies (int64)
10. **WHEEZING**: Wheezing occurrence (int64)
11. **ALCOHOL CONSUMING**: Alcohol consumption habit (int64)
12. **COUGHING**: Coughing occurrence (int64)
13. **SHORTNESS OF BREATH**: Shortness of breath occurrence (int64)
14. **SWALLOWING DIFFICULTY**: Difficulty in swallowing (int64)
15. **CHEST PAIN**: Chest pain occurrence (int64)
16. **LUNG_CANCER**: Lung cancer status (object)

----
## Step 2: Summary Statistics and Initial Exploration

### Displaying Summary Statistics of Numerical Features
In this step, we calculate and display summary statistics for the numerical features in the dataset to understand their distributions and key statistics.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('SurveyLungCancer.csv')

# Display summary statistics of numerical features
print("\nSummary Statistics of Numerical Features:")
print(data.describe())

# Display the first few rows of the dataset
print("\nFirst Few Rows of the Dataset:")
print(data.head())
```

### Output Description
The output provides a summary of key statistics for each numerical feature, as well as the first few rows of the dataset for a quick inspection:

```
Summary Statistics of Numerical Features:
              AGE     SMOKING  YELLOW_FINGERS     ANXIETY  PEER_PRESSURE  \
count  309.000000  309.000000      309.000000  309.000000     309.000000   
mean    62.673139    1.563107        1.569579    1.498382       1.501618   
std      8.210301    0.496806        0.495938    0.500808       0.500808   
min     21.000000    1.000000        1.000000    1.000000       1.000000   
25%     57.000000    1.000000        1.000000    1.000000       1.000000   
50%     62.000000    2.000000        2.000000    1.000000       2.000000   
75%     69.000000    2.000000        2.000000    2.000000       2.000000   
max     87.000000    2.000000        2.000000    2.000000       2.000000   

       CHRONIC DISEASE    FATIGUE     ALLERGY     WHEEZING  ALCOHOL CONSUMING  \
count       309.000000  309.000000  309.000000  309.000000         309.000000   
mean          1.504854    1.673139    1.556634    1.556634           1.556634   
std           0.500787    0.469827    0.497588    0.497588           0.497588   
min           1.000000    1.000000    1.000000    1.000000           1.000000   
25%           1.000000    1.000000    1.000000    1.000000           1.000000   
50%           2.000000    2.000000    2.000000    2.000000           2.000000   
75%           2.000000    2.000000    2.000000    2.000000           2.000000   
max           2.000000    2.000000    2.000000    2.000000           2.000000   

         COUGHING  SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  
count  309.000000           309.000000             309.000000  309.000000  
mean     1.579288             1.640777               1.469256    1.556634  
std      0.494474             0.480551               0.499863    0.497588  
min      1.000000             1.000000               1.000000    1.000000  
25%      1.000000             1.000000               1.000000    1.000000  
50%      2.000000             2.000000               1.000000    2.000000  
75%      2.000000             2.000000               2.000000    2.000000  
max      2.000000             2.000000               2.000000    2.000000  

First Few Rows of the Dataset:
  GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \
0      M   69        1               2        2              1   
1      M   74        2               1        1              1   
2      F   59        1               1        1              2   
3      M   63        2               2        2              1   
4      F   63        1               2        1              1   

   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \
0                1         2         1         2                  2         2   
1                2         2         2         1                  1         1   
2                1         2         1         2                  1         2   
3                1         1         1         1                 

 2         1   
4                1         1         1         2                  1         2   

   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN LUNG_CANCER  
0                    2                      2           2         YES  
1                    2                      2           2         YES  
2                    2                      1           2          NO  
3                    1                      2           2          NO  
4                    2                      1           1          NO  
```

This step gives an overview of the statistical distribution of the numerical features and helps to identify any potential anomalies or areas of interest in the data. Additionally, viewing the first few rows provides a quick sense of the data format and feature values.

---

## Step 3: Data Cleaning

### Handling Missing Values
In this step, we check for missing values in the dataset. The code checks for missing values in each column and prints the count of missing values. As there are no missing values in the dataset, no further action is required for handling missing values.

```python
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)
```

### Handling Outliers
Outliers are detected and removed using the Interquartile Range (IQR) method. The code defines a function to calculate the IQR and identify outliers for each relevant column. Outliers are then removed from the dataset, and the cleaned dataset is saved to a new CSV file.

```python
# Define a function to calculate IQR and identify outliers
def detect_and_remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Removing outliers
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df_cleaned

# Apply the function to each relevant column
columns_to_check = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE']

for column in columns_to_check:
    df = detect_and_remove_outliers(data, column)

# Save the cleaned dataset
df.to_csv('cleaned LungCancer_dataset.csv', index=False)

print("Outliers removed and cleaned dataset saved as 'cleaned LungCancer_dataset.csv'.")
```

### Output Description
The code outputs the count of missing values and a message confirming the removal of outliers and saving of the cleaned dataset.

--
## Step 4: Importing and Exploring the Cleaned Dataset

To work with the cleaned dataset, we need to import it into our Python environment using pandas. Follow these steps to import and explore the cleaned dataset:

1. Import the pandas library:

    ```python
    import pandas as pd
    ```

2. Load the cleaned dataset:

    ```python
    cleaned_data = pd.read_csv('Finalcleaned_LungCancer_dataset.csv')
    ```

3. Display the first few rows of the cleaned dataset:

    ```python
    print("First Few Rows of the Cleaned Dataset:")
    print(cleaned_data.head())
    ```

4. Display summary statistics of the cleaned dataset:

    ```python
    print("\nSummary Statistics of the Cleaned Dataset:")
    print(cleaned_data.describe())
    ```

This will help us understand the structure and distribution of the data, which is essential for further analysis and modeling.
--
## Step 5: Data Encoding and Visualization

After cleaning the dataset, we perform encoding for categorical variables and visualize the data to gain insights.

1. Import necessary libraries:

    ```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    ```

2. Load the cleaned data:

    ```python
    cleaned_data = pd.read_csv('Finalcleaned_LungCancer_dataset.csv')
    ```

3. Initialize LabelEncoder and encode categorical variables:

    ```python
    label_encoder = LabelEncoder()
    cleaned_data['GENDER'] = label_encoder.fit_transform(cleaned_data['GENDER'])
    cleaned_data['LUNG_CANCER'] = label_encoder.fit_transform(cleaned_data['LUNG_CANCER'])
    ```

4. Save the updated dataset to a new file:

    ```python
    cleaned_data.to_csv('Finalcleaned_LungCancer_dataset.csv', index=False)
    ```

5. Check the updated data types:

    ```python
    print(cleaned_data.dtypes)
    ```

6. Load the final cleaned data for visualization:

    ```python
    final_cleaned_data = pd.read_csv('Finalcleaned_LungCancer_dataset.csv')
    ```

7. Histograms for numerical features:

    ```python
    final_cleaned_data.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()
    ```

8. Count plots for categorical variables:

    ```python
    plt.figure(figsize=(10, 6))
    sns.countplot(x='GENDER', data=final_cleaned_data)
    plt.title('Count of Gender')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='LUNG_CANCER', data=final_cleaned_data)
    plt.title('Count of Lung Cancer')
    plt.show()
    ```

9. Correlation heatmap for numeric features:

    ```python
    plt.figure(figsize=(12, 8))
    numeric_data = final_cleaned_data.select_dtypes(include=['number'])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
    ```

These steps provide an overview of data encoding and visualization, aiding in understanding the dataset's characteristics and relationships between variables.
--

## Step 6: Correlation Heatmap for Numeric Features

To understand the relationship between different features and the target variable (LUNG_CANCER), we create a correlation heatmap.

1. Correlation Heatmap for Numeric Features:

    ```python
    # Correlation Heatmap for Numeric Features
    plt.figure(figsize=(12, 8))
    # Selecting only numeric columns for correlation calculation
    numeric_data = cleaned_data.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    correlation_matrix = numeric_data.corr()

    # Extract correlation values for LUNG_CANCER
    lung_cancer_corr = correlation_matrix['LUNG_CANCER']

    # Create a heatmap for the correlation values of LUNG_CANCER with other features
    plt.figure(figsize=(8, 6))
    sns.heatmap(lung_cancer_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
    plt.title('Correlation of LUNG_CANCER with Other Features')
    plt.show()
    ```

This heatmap helps us identify which features are most correlated with the presence of lung cancer, providing valuable insights for further analysis.
---
## Step 7: Calculate Correlation Coefficients with LUNG_CANCER

To further explore the relationship between the target variable (LUNG_CANCER) and other features, we calculate correlation coefficients and visualize the top correlated variables.

1. Calculate correlation coefficients between 'LUNG_CANCER' and other variables:

    ```python
    # Calculate correlation coefficients between 'LUNG_CANCER' and other variables
    correlation_with_lung_cancer = cleaned_data.corr()['LUNG_CANCER']

    # Sort the correlation coefficients by their absolute values
    sorted_correlation = correlation_with_lung_cancer.abs().sort_values(ascending=False)

    # Extract top 5 variables (excluding 'LUNG_CANCER' itself)
    top_variables = sorted_correlation[1:6]

    # Create a bar plot to visualize the correlation coefficients
    plt.figure(figsize=(10, 6))
    top_variables.plot(kind='bar', color='skyblue')
    plt.title('Top 5 Correlated Variables with LUNG_CANCER')
    plt.xlabel('Variables')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    ```

2. Display the top 5 correlated features and their correlation values:

    ```python
    # Display the top 5 correlated features and their correlation values
    print("Top 5 features most correlated with LUNG_CANCER:")
    print(top_variables)
    ```

This analysis helps us identify the most influential factors related to the presence of lung cancer in the dataset.
--

## Step 8: Load and Check the Cleaned Data

After performing various data processing steps, we reload the cleaned dataset and inspect its structure.

1. Load the cleaned data:

    ```python
    # Load the cleaned data
    cleaned_data = pd.read_csv('Finalcleaned_LungCancer_dataset.csv')
    ```

2. Display information about the cleaned dataset:

    ```python
    print("\n1. Finalcleaned_LungCancer_dataset Info:")
    print(cleaned_data.info())
    ```

This step ensures that the data is properly loaded and provides an overview of its structure, including the number of entries and data types.
--
## Step 9: Feature Engineering and Correlation Analysis

In this step, we perform feature engineering to create a new feature 'HEALTH_RISK_SCORE' and conduct correlation analysis to identify the top correlated variables with 'LUNG_CANCER'.

1. Import necessary libraries:

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    ```

2. Load the cleaned data:

    ```python
    # Load the cleaned data
    cleaned_data = pd.read_csv('Finalcleaned_LungCancer_dataset.csv')
    ```

3. Print out the list of columns in the dataset:

    ```python
    print("Columns in the dataset:")
    print(cleaned_data.columns)
    ```

4. Feature Engineering: Create a new feature 'HEALTH_RISK_SCORE':

    ```python
    existing_columns = ['SMOKING', 'ALCOHOL CONSUMING', 'CHRONIC DISEASE', 'ANXIETY', 'PEER_PRESSURE']
    cleaned_data['HEALTH_RISK_SCORE'] = cleaned_data[existing_columns].sum(axis=1)
    ```

5. Correlation Matrix and Analysis:

    ```python
    # Correlation Matrix
    correlation_matrix = cleaned_data.corr()

    # Calculate correlation coefficients between 'LUNG_CANCER' and other variables
    correlation_with_lung_cancer = correlation_matrix['LUNG_CANCER']

    # Sort the correlation coefficients by their absolute values
    sorted_correlation = correlation_with_lung_cancer.abs().sort_values(ascending=False)

    # Extract top 5 variables (excluding 'LUNG_CANCER' itself)
    top_variables = sorted_correlation[1:6]

    # Create a bar plot to visualize the correlation coefficients
    plt.figure(figsize=(10, 6))
    top_variables.plot(kind='bar', color='skyblue')
    plt.title('Top 5 Correlated Variables with LUNG_CANCER')
    plt.xlabel('Variables')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Display the top 5 correlated features and their correlation values
    print("Top 5 features most correlated with LUNG_CANCER:")
    print(top_variables)
    ```

6. Save the modified dataset:

    ```python
    # Save the modified dataset
    cleaned_data.to_csv('Finalcleaned_LungCancer_dataset_with_HEALTH_RISK_SCORE.csv', index=False)
    ```

This step enhances our dataset by introducing a new feature and provides insights into the variables most correlated with lung cancer.
---
## Step 10: Model Training and Evaluation

In this step, we train and evaluate machine learning models using the dataset to predict lung cancer.

1. Import necessary libraries:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    ```

2. Split the data into features and target variable:

    ```python
    X = cleaned_data[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                      'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 
                      'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'HEALTH_RISK_SCORE']]
    y = cleaned_data['LUNG_CANCER']
    ```

3. Split the data into training and testing sets:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. Initialize models:

    ```python
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    ```

5. Train and evaluate each model:

    ```python
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy - {accuracy}")
    ```

This step helps us assess the performance of different machine learning algorithms in predicting lung cancer.
---
## Step 11: Hyperparameter Tuning with Grid Search

In this step, we perform hyperparameter tuning using Grid Search with Cross-Validation to find the optimal parameters for the Random Forest classifier.

1. Import necessary libraries:

    ```python
    from sklearn.model_selection import GridSearchCV
    ```

2. Define the parameter grid for Random Forest:

    ```python
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    ```

3. Initialize Random Forest classifier:

    ```python
    rf_clf = RandomForestClassifier()
    ```

4. Perform Grid Search CV:

    ```python
    grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    ```

5. Get the best parameters and best score:

    ```python
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    ```

This step helps us find the best hyperparameters for the Random Forest classifier, improving its performance in predicting lung cancer.
---
## Step 12: Training and Evaluating Model with Relevant Features

In this step, we train and evaluate the Random Forest classifier using only relevant features identified through previous analysis.

1. Import necessary libraries:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ```

2. Define the relevant features:

    ```python
    relevant_features = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE','HEALTH_RISK_SCORE']
    ```

3. Extract relevant features and target variable:

    ```python
    X_relevant = cleaned_data[relevant_features]
    y = cleaned_data['LUNG_CANCER']
    ```

4. Split the data into training and testing sets:

    ```python
    X_train_relevant, X_test_relevant, y_train, y_test = train_test_split(X_relevant, y, test_size=0.2, random_state=42)
    ```

5. Train Random Forest with best parameters:

    ```python
    from sklearn.ensemble import RandomForestClassifier

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
    rf_clf.fit(X_train_relevant, y_train)
    ```

6. Predict on the test set and evaluate the model:

    ```python
    y_pred = rf_clf.predict(X_test_relevant)

    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest (with relevant features) Accuracy:", accuracy)
    ```

This step demonstrates the training and evaluation of the Random Forest classifier using only relevant features, improving efficiency and model performance.
---
## Step 13: Calculating Precision and Recall

In this step, we calculate precision and recall for the Random Forest classifier using relevant features and evaluate if they meet specified criteria.

1. Import necessary libraries:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score
    from sklearn.ensemble import RandomForestClassifier
    ```

2. Define the relevant features:

    ```python
    relevant_features = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'HEALTH_RISK_SCORE']
    ```

3. Extract relevant features and target variable:

    ```python
    X_relevant = cleaned_data[relevant_features]
    y = cleaned_data['LUNG_CANCER']
    ```

4. Split the data into training and testing sets:

    ```python
    X_train_relevant, X_test_relevant, y_train, y_test = train_test_split(X_relevant, y, test_size=0.2, random_state=42)
    ```

5. Train Random Forest with best parameters:

    ```python
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
    rf_clf.fit(X_train_relevant, y_train)
    ```

6. Predict on the test set:

    ```python
    y_pred = rf_clf.predict(X_test_relevant)
    ```

7. Calculate precision and recall:

    ```python
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    ```

8. Print precision and recall:

    ```python
    print("Precision:", precision)
    print("Recall:", recall)
    ```

9. Check if precision and recall meet the specified criteria:

    ```python
    if precision >= 0.3 and recall >= 0.3:
        print("Both precision and recall meet the criteria.")
    else:
        print("Precision and/or recall do not meet the criteria.")
    ```

This step evaluates the performance of the Random Forest classifier based on precision and recall, ensuring the model meets the specified criteria.
---
## Step 14: Building and Saving a Machine Learning Pipeline

In this step, we build and save a machine learning pipeline that includes data preprocessing and model training.

1. Import necessary libraries:

    ```python
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    ```

2. Load the dataset:

    ```python
    df = pd.read_csv('Finalcleaned_LungCancer_dataset_with_HEALTH_RISK_SCORE.csv')
    ```

3. Separate target and features:

    ```python
    X = df.drop(columns=['LUNG_CANCER'])  # Exclude the target variable
    y = df['LUNG_CANCER']
    ```

4. Split the data into training and testing sets:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

5. Define the column transformer for preprocessing:

    ```python
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['AGE', 'HEALTH_RISK_SCORE']),  # Standardize numeric features
            # No categorical features in this case
        ], remainder='passthrough'  # Passthrough other features
    )
    ```

6. Define the model:

    ```python
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
    ```

7. Create the pipeline:

    ```python
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    ```

8. Fit the pipeline:

    ```python
    pipeline.fit(X_train, y_train)
    ```

9. Save the fitted pipeline:

    ```python
    joblib.dump(pipeline, 'lllUNGCANCER_trained_model_pipeline.joblib')
    ```

This step creates and saves a pipeline that includes preprocessing and model training steps, making it convenient for future use.
---
## Step 15: Deploying Lung Cancer Prediction Interface

In this final step, we deploy an interface using Gradio, allowing users to interactively predict whether a person is likely to have lung cancer based on their input.

1. Import necessary libraries:

    ```python
    import pandas as pd
    import joblib
    import gradio as gr
    ```

2. Load the trained model pipeline:

    ```python
    model_pipeline = joblib.load('lllUNGCANCER_trained_model_pipeline.joblib')
    ```

3. Define the prediction function:

    ```python
    def predict_lung_cancer(age, sex, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain):
        # Code for prediction
    ```

4. Create the Gradio interface:

    ```python
    interface = gr.Interface(
        fn=predict_lung_cancer,
        inputs=[
            # Input components
        ],
        outputs=gr.Textbox(label="Lung Cancer Prediction"),
        title="Lung Cancer Prediction",
        description="Predict whether a person is likely to have Lung Cancer."
    )
    ```

5. Launch the Gradio interface:

    ```python
    interface.launch()
    ```

This step deploys an interactive interface using Gradio, enabling users to input data and receive predictions regarding lung cancer.
---
