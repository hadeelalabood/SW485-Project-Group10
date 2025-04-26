#!/usr/bin/env python
# coding: utf-8

# # "Career Recommendation system jupyter notebook-Group10"

# # (Phase 1: Problem Understanding and Data Exploration)

# 
# 
# 
# 
# # 1- Goal of the Dataset
# 
# The purpose of collecting this dataset is to develop a **career recommendation system** that helps job seekers find suitable roles by analyzing their past experiences, skills, and preferences. This system aims to assist job seekers in identifying the most relevant job opportunities while also supporting recruiters in finding the best candidates for their requirements.
# 
# ### Purpose of Collecting the Dataset
# This dataset was collected for the purpose of **requirements classification**, where the goal is to:
# 1. **Analyze Job Requirements**: Understand the skills, experiences, and qualifications required for different roles.
# 2. **Evaluate Candidate Profiles**: Identify the suitability of candidates for specific job positions based on their profiles.
# 3. **Perform Skill Gap Analysis**: Recommend skills or certifications to help candidates align with the requirements of their desired roles.
# 
# ### Objectives of the Dataset
# 1. **Personalized Job Recommendations**: Match job postings with candidates' qualifications and aspirations.
# 2. **Career Path Insights**: Provide recommendations for career progression based on historical job roles and skills.
# 3. **Skill-Gap Analysis**: Identify missing qualifications to achieve desired job roles.
# 4. **Improving Recruitment Efficiency**: Assist employers in finding well-suited candidates.
# 
# By leveraging **supervised and unsupervised learning techniques**, this dataset enables the development of an intelligent job recommendation system that improves job search efficiency and enhances the overall recruitment experience for both job seekers and employers.
# 

# # 2- Source of the dataset
# The source of dataset: https://github.com/boratonAJ/Job-Recommendation-System/blob/master/datasets/data_science_extract.csv

# # 3- General Information

# ### 1. Load and Preview the Dataset

# In[1]:


import pandas as pd
from scipy.sparse import hstack


df=pd.read_csv("data_science_extract.csv")  # Load dataset
df.head()  # Show first 5 rows


# ### 2. Get Basic Information

# In[2]:


df.info()  # Provides column data types and missing values


# we can see that the dataset consist of 2000 rows, and 9 columns/variables, and for the types we can see 2 varibles has int64 type and 7 variables has object type. 

# ### 3. Get Dataset Shape (Rows & Columns Count)

# In[3]:


print(f"Number of observations (rows): {df.shape[0]}")
print(f"Number of variables (columns): {df.shape[1]}")


# ### 4. Check Data Types and Unique Classes (for Categorical Data)

# In[4]:


print("Column Data Types:")
print(df.dtypes)

# Find unique values in categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())


# ### 5. Summary Statistics

# For numerical varibles: 

# In[5]:


df.describe()  # Provides count, mean, std, min, max, percentiles


# For categorical variables:

# In[6]:


df.describe(include=['object'])  # Summary of categorical data


# ### 6. Check Class Distribution (for Classification Problems)

# Check Class Distribution:

# In[7]:


target_column = "careerjunction_za_primary_jobtitle"  # Set target column

# Count occurrences of each class
class_distribution = df[target_column].value_counts()

# Display top 10 most frequent classes
print(class_distribution.head(10))


# Check Class Distribution in Percentage:
# ###### To see the percentage of each job title:

# In[8]:


print(class_distribution.head(10) / len(df) * 100)


# In[9]:


# Plot bar chart for class distribution
import pandas as pd
import matplotlib.pyplot as plt

# Select top 10 most common classes for visualization
class_distribution = df[target_column].value_counts()
top_classes = class_distribution.head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_classes.index, top_classes.values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Job Titles")
plt.ylabel("Count")
plt.title("Top 10 Most Frequent Job Titles in Dataset")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # 4- Summary of the dataset

# In[10]:


import pandas as pd

df = pd.read_csv("data_science_extract.csv") 

print("Sample of the dataset:")
display(df.head())




# In[11]:


# Analyze missing values 
missing_values = df.isnull().sum()
print("\nMissing Values Per Column:")
print(missing_values[missing_values > 0])  # Only columns containing missing values


# In[12]:


#Basic analysis for data
#Extract basic statistics such as mean and frequency
print("\nStatistical Summary:")
print(df.describe(include="all"))  


# In[13]:


import matplotlib.pyplot as plt

#Analyze number of odd values in first 5 columns  
unique_counts = df.iloc[:, :5].nunique()

#Draw a bar chart representing the number of unique values per column
plt.figure(figsize=(10, 5))
unique_counts.plot(kind="bar", color="skyblue")
plt.xlabel("Columns")
plt.ylabel("Unique Values Count")
plt.title("Unique Values per Column")
plt.xticks(rotation=45)
plt.show()


# In[14]:


import pandas as pd
from IPython.display import display

#Create a table containing the number of unique values for each column
unique_values_table = pd.DataFrame({
    "Column": df.columns,
    "Unique Values Count": df.nunique()
})

#Display table
display(unique_values_table)


# # 5- Preprocessing techniques

# ### 1. Load and Inspect the Data

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# #### Check data *before* preprocessing

# In[16]:


# Check basic info
df.info()


# In[17]:


# Display first few rows
df.head()


# In[18]:


# Display last few rows
df.tail()


# ### 2. Handle Missing Values

# In[19]:


# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)


# In[20]:


#find percentage of missing value for each column 
df.isnull().sum()/df.shape[0]*100


# We can see that there is no missing values in data, so no need for dropping columns or replace values.

# ### 3. Remove Duplicates 

# In[21]:


#finding duplicates
df.duplicated().sum()


# We can see that there is no duplicate indata, so no need for dropping columns.

# ### 4. Remove Unnecessary Columns

# In[22]:


# Display first few values of these columns
print(df[["Unnamed: 0", "id"]].head(10))


# Unamed and id columns seem useless as they represent identification of each row, so we will drop it.

# In[23]:


df.drop(columns=["Unnamed: 0", "id"], inplace=True)


# ### 5.Convert Text Data into Structured Format
# Since many columns contain lists as text (e.g., "['Software Developer', 'Engineer']"), we will clean them.

# In[24]:


import ast

# Convert string representations of lists into actual lists
for col in ["careerjunction_za_courses", "careerjunction_za_future_jobtitles",
            "careerjunction_za_historical_jobtitles", "careerjunction_za_recent_jobtitles",
            "careerjunction_za_skills"]:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


# ### 6. Encode Categorical Variables
# Convert categorical variables to numerical representations.

# In[25]:


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in ["careerjunction_za_primary_jobtitle"]:  # Encode only categorical target column
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])


# ### 7. Feature Engineering (Extract Key Features)
# If needed, create new features from existing ones.

# In[26]:


#Extract the number of skills listed for each individual
df["num_skills"] = df["careerjunction_za_skills"].apply(lambda x: len(x) if isinstance(x, list) else 0)


# In[27]:


#Count the number of past job titles.
df["num_past_jobs"] = df["careerjunction_za_historical_jobtitles"].apply(lambda x: len(x) if isinstance(x, list) else 0)


# ### 8. Visualize Relationships
# Check relationships between key variables.

# ##### Job Titles vs. Number of Skills
# 

# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.boxplot(x="careerjunction_za_primary_jobtitle", y="num_skills", data=df)
plt.xticks(rotation=45, ha='right')
plt.title("Skills Count by Job Title")
plt.show()


# ### 9. Normalize Numerical Data

# we are using Min-Max Scaling (scales values between 0 and 1) method to normlize numerical data

# In[29]:


# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(numerical_cols)


# In[30]:


from sklearn.preprocessing import MinMaxScaler

# Select only numerical columns
numerical_cols = ['num_skills', 'num_past_jobs']


# In[31]:


# Initialize and apply scaler
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# ### 10. Remove outliers
# First step: Detect Outliers Using the IQR Method (Interquartile Range)

# In[32]:


import numpy as np

# Function to detect outliers using IQR
def detect_outliers_iqr(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    return outliers

# Select numerical columns after normalization
numerical_cols = ['num_skills', 'num_past_jobs']

# Detect outliers
outliers = detect_outliers_iqr(df, numerical_cols)

# Display the number of outliers per column
for col, outlier_data in outliers.items():
    print(f"{col}: {len(outlier_data)} outliers detected")


# Second step: Visualize Outliers Using Boxplots

# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot boxplots to visualize outliers
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


# we can see from the from the graph that there is outliers in data and we should remove them. 

# Third step:Remove Outliers

# In[34]:


def wisker(col):
    q1,q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw


# In[35]:


wisker(df['num_past_jobs'])


# In[36]:


for i in ['num_skills', 'num_past_jobs']:
    lw,uw=wisker(df[i])
    df[i]=np.where(df[i]<lw,lw,df[i])
    df[i]=np.where(df[i]>lw,lw,df[i])
    


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot boxplots to visualize outliers
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


# In[38]:


import numpy as np

# Function to detect outliers using IQR
def detect_outliers_iqr(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    return outliers

# Select numerical columns after normalization
numerical_cols = ['num_skills', 'num_past_jobs']

# Detect outliers
outliers = detect_outliers_iqr(df, numerical_cols)

# Display the number of outliers per column
for col, outlier_data in outliers.items():
    print(f"{col}: {len(outlier_data)} outliers detected")


# As we can see now there is no outliers.

# ### 11. Remove garbage values
# First step: detect grbage values

# In[39]:


# Identify unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Print unique values in each categorical column (only top 20 values)
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts().head(20))  # Adjust the number to see more if needed

    


# Second step: fix garbage values

# In[40]:


import pandas as pd

# Replace placeholders with NaN
garbage_placeholders = ["Unknown", "?", "N/A", "None", "null"]
df.replace(garbage_placeholders, pd.NA, inplace=True)

# Handle missing values column by column

# Fill numerical columns with median
for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with "Unknown"
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col].fillna("Unknown", inplace=True)

# Check if missing values are handled
print(df.isnull().sum())


# # Finally: show data after preprocessing

# In[41]:


# Display statistics for numerical columns
df.describe()


# In[42]:


print(df.info())


# In[43]:


df.head()


# In[44]:


df.describe(include="all")


# In[45]:


df.to_csv("processed_data.csv", index=False)


# In[46]:


import os
print(os.getcwd())  # Prints the current working directory


# # (Phase2: Build a Supervised Learning Model)

# 
# Our dataset involves career prediction based on historical job titles and skills. This is a classification problem where each data point (skills and job titles) is mapped to a future job title. The choice of Support Vector Machine (SVM) and Random Forest (RF) is justified based on the following factors:

# ## 1. Support Vector Machine (SVM)
# #### -Handles High-Dimensional Text Data Well: 
# Since we are using TF-IDF vectorization, the features are sparse and high-dimensional. SVM works well in such cases because it finds a hyperplane that maximizes class separation.
# #### -Good for Small to Medium Datasets: 
# Since our dataset consist of 2000 rows it consider small data set, and SVM is known to work well even with limited training data, making it suitable for our dataset.
# #### -Robust to Overfitting (with Linear Kernel):
# Since our features are textual representations, using a linear kernel prevents overfitting while maintaining interpretability.
# Works Well with Imbalanced Data: SVM is effective when there is class imbalance (some job titles appear much less frequently than others), as it focuses on the hardest-to-classify points near the decision boundary.

# ## 2. Random Forest (RF)
# #### -Handles Non-Linear Relationships Well: 
# Unlike SVM, which finds a linear decision boundary (unless using kernels), Random Forest can capture complex relationships between job history, skills, and job title.
# #### -Handles Categorical & Text Features Well: 
# Our dataset has job titles, skills, and employer names, which can be encoded as categorical data or word embeddings.
# #### -Resistant to Overfitting: 
# Since Random Forest averages multiple decision trees, it reduces the risk of overfitting.
# #### -Feature Importance: 
# It helps interpret which skills, courses, or past jobs are most important in career recommendations.
# #### -Works Well with Mixed Data: 
# It handles both numerical (num_skills, num_past_jobs) and categorical (careerjunction_za_skills, careerjunction_za_historical_jobtitles) data efficiently.

# 
# 
# -----------------------------------------------------------------------------------------------------------------------
# Now, let's implement both two algorithms and compare results:

# In[49]:


# Import necessary libraries
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack


# #### 1. LOAD AND CLEAN DATASET

# In[50]:


# Load dataset
file_path = 'data_science_extract(in).csv'
df = pd.read_csv(file_path, encoding="latin1", usecols=[
    "careerjunction_za_future_jobtitles",
    "careerjunction_za_skills",
    "careerjunction_za_historical_jobtitles"
])


# In[51]:


# Drop rows where future job title is missing or empty
df.dropna(subset=['careerjunction_za_future_jobtitles'], inplace=True)
df = df[df['careerjunction_za_future_jobtitles'].str.strip() != "[]"]


# In[52]:


# Standardize target labels (strip whitespace, fix inconsistencies)
df['careerjunction_za_future_jobtitles'] = df['careerjunction_za_future_jobtitles'].str.strip()
df['careerjunction_za_future_jobtitles'] = df['careerjunction_za_future_jobtitles'].str.replace('\xa0', ' ', regex=False)


# In[53]:


# Convert skills & job titles from string representations of lists to actual lists
df['careerjunction_za_skills'] = df['careerjunction_za_skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df['careerjunction_za_historical_jobtitles'] = df['careerjunction_za_historical_jobtitles'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])


# In[54]:


# Convert lists to text format
df['skills_text'] = df['careerjunction_za_skills'].apply(lambda skills: ' '.join(skills))
df['hist_text'] = df['careerjunction_za_historical_jobtitles'].apply(lambda jobs: ' '.join(jobs))


# ####  2. MERGE RARE CLASSES INTO "OTHER"

# In[55]:


# Count occurrences of each job title category
class_counts = df['careerjunction_za_future_jobtitles'].value_counts()


# In[56]:


# Define threshold (minimum samples per job title category)
min_samples = 2


# In[57]:


# Replace rare categories with "Other"
df['careerjunction_za_future_jobtitles'] = df['careerjunction_za_future_jobtitles'].apply(
    lambda x: x if class_counts[x] >= min_samples else "Other"
)


# #### 3. SPLIT DATA INTO TRAIN & TEST SETS
# 

# In[58]:


# Define features (skills_text + hist_text) and target
X = df[['skills_text', 'hist_text']]
y = df['careerjunction_za_future_jobtitles']


# In[59]:


# Stratified train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# #### 4. HANDLE CLASS IMBALANCE (OVERSAMPLING)

# In[61]:


# Combine X_train and y_train to resample minority classes
train_data = X_train.copy()
train_data['future_jobtitle'] = y_train.values


# In[62]:


# Find the max count of the most common class
max_count = train_data['future_jobtitle'].value_counts().max()


# In[63]:


#Perform oversampling for minority classes
oversampled_train_parts = []
for category, group in train_data.groupby('future_jobtitle'):
   if len(group) < max_count:
       group_oversampled = resample(group, replace=True, n_samples=max_count, random_state=42)
       oversampled_train_parts.append(group_oversampled)
   else:
       oversampled_train_parts.append(group)


# In[64]:


# Create new balanced training set
train_data_balanced = pd.concat(oversampled_train_parts).reset_index(drop=True)
train_data_balanced = train_data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


# In[65]:


# Extract oversampled features and labels
X_train_balanced = train_data_balanced[['skills_text', 'hist_text']]
y_train_balanced = train_data_balanced['future_jobtitle']


# #### 5. TEXT FEATURE EXTRACTION (TF-IDF)

# In[66]:


# Initialize TF-IDF vectorizers for skills & job titles
skills_vectorizer = TfidfVectorizer(max_features=5000)
history_vectorizer = TfidfVectorizer(max_features=5000)


# In[67]:


# Fit and transform on training data
X_train_skill_tfidf = skills_vectorizer.fit_transform(X_train_balanced['skills_text'])
X_train_hist_tfidf = history_vectorizer.fit_transform(X_train_balanced['hist_text'])


# In[68]:


# Transform test data
X_test_skill_tfidf = skills_vectorizer.transform(X_test['skills_text'])
X_test_hist_tfidf = history_vectorizer.transform(X_test['hist_text'])


# In[69]:


# Combine TF-IDF matrices
X_train_final = hstack([X_train_skill_tfidf, X_train_hist_tfidf])
X_test_final = hstack([X_test_skill_tfidf, X_test_hist_tfidf])


# #### 5. TEXT FEATURE EXTRACTION (TF-IDF)

# In[70]:


# Initialize TF-IDF vectorizers for skills & job titles
skills_vectorizer = TfidfVectorizer(max_features=5000)
history_vectorizer = TfidfVectorizer(max_features=5000)


# In[71]:


# Fit and transform on training data
X_train_skill_tfidf = skills_vectorizer.fit_transform(X_train_balanced['skills_text'])
X_train_hist_tfidf = history_vectorizer.fit_transform(X_train_balanced['hist_text'])


# In[72]:


# Transform test data
X_test_skill_tfidf = skills_vectorizer.transform(X_test['skills_text'])
X_test_hist_tfidf = history_vectorizer.transform(X_test['hist_text'])


# In[73]:


# Combine TF-IDF matrices
X_train_final = hstack([X_train_skill_tfidf, X_train_hist_tfidf])
X_test_final = hstack([X_test_skill_tfidf, X_test_hist_tfidf])


#  #### 6. TRAIN & EVALUATE MACHINE LEARNING MODELS

# In[74]:


# Train SVM Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_final, y_train_balanced)


# In[75]:


# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_final, y_train_balanced)


# #### 7. MODEL PERFORMANCE EVALUATION

# In[76]:


# Predict on the test set
y_pred_svm = svm_model.predict(X_test_final)
y_pred_rf = rf_model.predict(X_test_final)


# In[77]:


# Print Accuracy Scores
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# In[78]:


# Print Classification Reports
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))


# In[79]:


print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))


# Based on the results, Random Forest (53.97%) outperformed SVM (49.77%) in accuracy, making it the better choice for predicting future job titles from skills and historical job data. The main reason for Random Forest’s superior performance is its ability to handle non-linear relationships better than SVM. Career progression is often complex and not strictly linear, and Random Forest, being an ensemble of decision trees, can capture these variations more effectively. Additionally, Random Forest is more robust to noise and outliers, as it averages across multiple trees, reducing the impact of misclassified instances. On the other hand, SVM relies on a linear decision boundary, which may not be the best fit for this type of data. Moreover, even after oversampling, some job categories may still be underrepresented, and Random Forest’s ability to handle imbalanced data through multiple decision paths gives it an advantage over SVM, which tends to struggle with class imbalance.

# ### Reasons for the Obtained Accuracy Values : 
# #### -Class Imbalance in the Dataset:
# The dataset contains a disproportionate number of samples for different job categories. Some job titles, such as "Software Development," have significantly more occurrences compared to others like "Finance & Accounting" or "Learning & Development." This imbalance causes the model to favor predicting the majority classes while failing to correctly classify underrepresented job titles.
# 
# #### -Overlapping Job Titles and Skills:
# Many job roles share similar skill sets, making it difficult for the model to distinguish between them. For example, "Data Analysis & Business Intelligence" and "Database Administration & Development" both require database knowledge, which confuses the model and leads to misclassifications.
# 
# #### -Limited Feature Representation Using TF-IDF:
# The model relies on TF-IDF (Term Frequency-Inverse Document Frequency) to represent job skills in a numerical format. However, TF-IDF does not capture semantic relationships between words. For instance, it treats "Machine Learning" and "Artificial Intelligence" as completely separate terms, even though they are conceptually related. This limitation reduces the model’s ability to generalize well.
# 
# #### -Insufficient Training Samples for Some Categories:
# Certain job categories have very few samples in the dataset, leading to zero recall for those classes. Since the model does not encounter enough examples of rare job titles, it struggles to make correct predictions, which is evident in categories like "Administrative & Support" and "Finance & Accounting."
# 
# #### -SVM Struggles with Complex Class Boundaries:
# Support Vector Machines (SVM) perform best when there is a clear separation between classes. However, due to overlapping skills and job roles in the dataset, SVM fails to define accurate decision boundaries, leading to lower accuracy.
# 
# #### -Random Forest Performs Better but is Still Biased:
# Random Forest captures non-linear patterns better than SVM, which is why it achieves higher accuracy. However, it still struggles with underrepresented job categories and tends to favor dominant job titles, causing lower recall for less common roles.
# 
# #### -Low Macro Average F1-Score Due to Imbalance:
# The overall performance of the model is negatively affected by the imbalance in job titles, leading to a low macro-average F1-score. The model performs well on frequently occurring job roles but fails to generalize well to less common job categories.
# 
# ------------------------------------------------------------------------------------------------------------------------
# 

# #### 8. FUNCTION FOR USER INPUT & PREDICTION

# In[80]:


def predict_future_job(skills, historical_jobs, model_choice='svm'):
    """Predict future job title based on input skills and historical job titles."""

    # Convert input lists to text
    skills_text = ' '.join(skills)
    hist_text = ' '.join(historical_jobs)

    # Transform input using trained TF-IDF vectorizers
    skills_tfidf = skills_vectorizer.transform([skills_text])
    hist_tfidf = history_vectorizer.transform([hist_text])

    # Combine feature vectors
    input_features = hstack([skills_tfidf, hist_tfidf])

    # Predict using selected model
    if model_choice == 'svm':
        prediction = svm_model.predict(input_features)
    elif model_choice == 'rf':
        prediction = rf_model.predict(input_features)
    else:
        raise ValueError("Invalid model choice. Choose 'svm' or 'rf'.")

    return prediction[0]


# In[81]:


print(predict_future_job(["Python", "Machine Learning"], ["Data Analyst"], model_choice='svm'))


# # (Phase3: Apply Unsupervised Learning)

# In this phase of the project, we focus on applying unsupervised learning techniques, specifically clustering, 
# to uncover hidden patterns and groupings in the data. Clustering allows us to segment similar data points without 
# relying on labeled outputs, which can be especially useful in recommendation systems where user preferences
#  or item categories are not always clearly defined.
# 
# The goal is to enhance the performance of our previous model by exploring whether meaningful clusters can guide
#  or refine the recommendations. For instance, grouping users with similar behavior or preferences could help 
#  suggest more relevant items to new or existing users based on their cluster.
# 
# We applied two clustering algorithms:
# - K-Means Clustering
# - Agglomerative Clustering
# 
# These algorithms were selected because they represent two different approaches: partitioning-based and 
# hierarchy-based clustering. Each was tested on the dataset after removing the class label to ensure a purely
# unsupervised analysis.

# In[82]:


import pandas as pd

# Load the dataset
file_path = "data_science_extract(in) (2).csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Keep only relevant columns
columns_to_keep = [
    'careerjunction_za_skills',
    'careerjunction_za_historical_jobtitles',
    'careerjunction_za_future_jobtitles'
]
df_cleaned = df[columns_to_keep].copy()

# Drop rows with missing skills or job history
df_cleaned.dropna(subset=['careerjunction_za_skills', 'careerjunction_za_historical_jobtitles'], inplace=True)

# Combine skills and historical job titles into one text column
df_cleaned['combined_text'] = (
    df_cleaned['careerjunction_za_skills'].astype(str) + ' ' +
    df_cleaned['careerjunction_za_historical_jobtitles'].astype(str)
)

# Clean formatting like [u'...'] and commas
df_cleaned['combined_text'] = df_cleaned['combined_text'].str.replace(r"[\[\]u'\",]", '', regex=True)

# Show a sample
df_cleaned[['combined_text', 'careerjunction_za_future_jobtitles']].head()


# In[83]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Convert combined text to numeric format using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(df_cleaned['combined_text'])

# Show shape of the TF-IDF matrix
X_tfidf.shape


# In[84]:


from sklearn.cluster import KMeans

# Apply KMeans with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_tfidf)

# Add cluster labels to the dataframe
df_cleaned['kmeans_cluster'] = clusters_kmeans

# Show sample with KMeans cluster
df_cleaned[['combined_text', 'careerjunction_za_future_jobtitles', 'kmeans_cluster']].head()


# In[85]:


from sklearn.cluster import AgglomerativeClustering

# Apply Agglomerative Clustering with the same number of clusters (10)
agglo = AgglomerativeClustering(n_clusters=10)
clusters_agglo = agglo.fit_predict(X_tfidf.toarray())

# Add the results to the dataframe
df_cleaned['agglo_cluster'] = clusters_agglo

# Show a sample
df_cleaned[['combined_text', 'careerjunction_za_future_jobtitles', 'agglo_cluster']].head()


# In[86]:


from sklearn.metrics import silhouette_score

# Calculate Silhouette Scores
sil_kmeans = silhouette_score(X_tfidf, df_cleaned['kmeans_cluster'])
sil_agglo = silhouette_score(X_tfidf, df_cleaned['agglo_cluster'])

print("KMeans Silhouette Score:", sil_kmeans)
print("Agglomerative Silhouette Score:", sil_agglo)


# In[87]:


# We tested k values from 2 to 10.
# Based on the silhouette score plot, k = 10 gave the highest score,
# indicating better-defined and well-separated clusters.
# That's why we used k = 10 in both KMeans and Agglomerative for fair comparison.

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Try multiple values of k and store silhouette scores
k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)
    score = silhouette_score(X_tfidf, labels)
    silhouette_scores.append(score)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', color='teal')
plt.title('Silhouette Score vs Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[88]:


from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())


# In[89]:


import matplotlib.pyplot as plt

# Plotting side-by-side
plt.figure(figsize=(14, 6))

# KMeans Clusters
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_cleaned['kmeans_cluster'], cmap='tab10', s=25)
plt.title("KMeans Clustering (k=10)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)

# Agglomerative Clusters
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_cleaned['agglo_cluster'], cmap='tab10', s=25)
plt.title("Agglomerative Clustering (k=10)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)

plt.tight_layout()
plt.show()


# ### Visual Comparison:
# 
# ### The PCA visualization shows the differences between KMeans and Agglomerative Clustering.
# 
# - **KMeans** produces more compact and slightly more separated clusters.
# - **Agglomerative** clustering results in more overlapping clusters, with less defined boundaries.
# 
# These observations align with the Silhouette Score results, where **KMeans scored higher**, indicating better cluster quality.

# In[90]:


# Show the most common future job titles per KMeans cluster
df_cleaned.groupby('kmeans_cluster')['careerjunction_za_future_jobtitles'].value_counts().groupby('kmeans_cluster').head(3)


# 
# ## Clustering Algorithms Used
# 
# 1. K-Means Clustering
# 
# K-Means is a popular partitioning algorithm that divides the dataset into k non-overlapping clusters. It works by:
# - Randomly initializing k centroids,
# - Assigning each data point to the nearest centroid,
# - Recalculating the centroids as the average of all points in each cluster,
# - Repeating the process until the centroids stop changing or reach a maximum number of iterations.
# 
# Why we used K-Means:
# 
# - Simple and fast, especially on large datasets.
# - Efficient when clusters are spherical and evenly sized.
# - Easy to interpret and visualize.
# 
# Performance on our dataset:
# - **Silhouette Coefficient:** 0.02218 
#   This low value suggests the clusters are not well-defined or are overlapping.
# 
# ---
# 
# 2. Agglomerative Clustering
# 
# Agglomerative Clustering is a type of hierarchical clustering that starts by treating each data point as its own
# cluster and merges the closest pairs step by step until only one cluster remains or a stopping condition is met.
# 
# Why we used Agglomerative Clustering:
# - Useful when we don’t know the optimal number of clusters in advance.
# - Can find nested patterns and works well with non-spherical data.
# - Doesn’t require centroid calculation, making it suitable for categorical or mixed data.
# 
# Performance on our dataset:
# - **Silhouette Coefficient:** 0.01209 
#   This is even lower than K-Means, indicating very weak separation between clusters.

# ## Best Algorithm: KMeans Clustering
# 
# | Reason | Explanation |
# |:------|:------------|
# | **Efficiency** | KMeans is very fast and scalable to large datasets, compared to algorithms like DBSCAN or Agglomerative clustering. |
# | **Simplicity** | Easy to implement and explain to non-technical audiences. |
# | **Works Well When Clusters are Spherical** | If data clusters are somewhat well-separated, KMeans performs very well. |
# | **Widely Used Standard** | KMeans is the industry standard for unsupervised clustering tasks. |
# | **Control over Number of Clusters** | You can specify the number of clusters (k) based on business needs or metrics like the Elbow method. |
# 
# ## Justification:
# After evaluating different unsupervised learning algorithms, we selected KMeans Clustering as the most suitable model for Phase 3. KMeans offers a highly efficient and scalable solution for identifying clusters within large datasets. It is simple to implement, easy to interpret, and allows control over the number of clusters through parameter tuning. KMeans performs well when clusters are well-separated and spherical, which matches the nature of our data. Moreover, it is a widely recognized industry standard for clustering tasks, ensuring reliability and proven effectiveness. Based on these advantages, KMeans was chosen as the final unsupervised model for this phase.

# ## Function for User Input & Prediction 

# In[10]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# STEP 1: Load your cleaned data
file_path = "data_science_extract(in) (2).csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Keep relevant columns and clean
columns_to_keep = ['careerjunction_za_skills', 'careerjunction_za_historical_jobtitles', 'careerjunction_za_future_jobtitles']
df_cleaned = df[columns_to_keep].dropna(subset=['careerjunction_za_skills', 'careerjunction_za_historical_jobtitles']).copy()
df_cleaned['combined_text'] = df_cleaned['careerjunction_za_skills'].astype(str) + ' ' + df_cleaned['careerjunction_za_historical_jobtitles'].astype(str)
df_cleaned['combined_text'] = df_cleaned['combined_text'].str.replace(r"[\[\]u\'\",]", '', regex=True)

# STEP 2: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(df_cleaned['combined_text'])

# STEP 3: Train KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
df_cleaned['kmeans_cluster'] = kmeans.fit_predict(X_tfidf)

# STEP 4: Define Recommendation Function
def recommend_future_jobs(user_input_text):
    # Transform the user input text
    user_input_vector = vectorizer.transform([user_input_text])
    
    # Predict the cluster
    predicted_cluster = kmeans.predict(user_input_vector)[0]
    
    # Get all possible future job titles from the predicted cluster
    future_jobs = df_cleaned[df_cleaned['kmeans_cluster'] == predicted_cluster]['careerjunction_za_future_jobtitles']
    
    # Return unique future job titles as recommendations
    return future_jobs.unique()

# STEP 5: Input from User
user_input = input("Enter your skills and previous job titles: ")
recommendations = recommend_future_jobs(user_input)

# STEP 6: Output Recommendations
print("\nRecommended Future Job Titles for You:")
for idx, job in enumerate(recommendations, start=1):
    print(f"{idx}. {job}")


# # (Phase4: Integrate Generative AI )

# In this phase we will integrate generitive AI for our dataset using GPT by applying two templates to provide career recommendation based on user input. Tampletes we are using will be Simple Instruction Prompt and Role-based Prompt.

# ## What generative AI we chose?
# We chose GPT as the generative AI model for our system due to its effectivness at interpreting complex input and producing context-aware responses that mimic expert-level advice. This aligns with our objective of providing detailed career recommendation based on user data like skills, courses, and job history. Compared to other models like LaMA, GPT offers higher-quality outputs without the need for extensive fine-tuning or local deployment, making it both powerful and practical for integration. Its ease of use through the OpenAI API, combined with its state-of-the-art performance in natural language generation, made GPT the most suitable choice for enhancing our system with intelligent and personalized career support.

# ## What tempalets we are using?
# In our system, we will apply two templates to GPT for generating recommendation advice. The first template is a Simple Instruction Prompt,this template is straightforward and useful for generating quick, general responses. The second template is a Role-Based Prompt, this approach is expected to yield more personalized and structured responses. We intend to compare the outcomes from both templates to determine which produces more relevant and actionable guidance.
# 

# ----------------------------------------------------------------
# 
# 
# Now, let's start integrating them for our dataset:

# ### 1-Install OpenAI Package
# 

# In[91]:


get_ipython().system('pip install openai')


# ### 2-Set Up API Key

# In[96]:


import openai

client = openai.OpenAI(api_key= "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")


# ### 3-Load and Prepare Data

# In[97]:


import pandas as pd

df = pd.read_csv("data_science_extract(in) (1).csv", encoding="ISO-8859-1")

sample = df.iloc[0]  # pick the first user
user_courses = sample["careerjunction_za_courses"]
user_skills = sample["careerjunction_za_skills"]
recent_jobs = sample["careerjunction_za_recent_jobtitles"]


# ### 4-Define Templates

# In[98]:


prompt_1 = f"Give career advice for someone with these skills: {user_skills}, and courses: {user_courses}."

prompt_2 = (
    f"You are a professional career advisor. Based on the following background:\n"
    f"Courses: {user_courses}\n"
    f"Skills: {user_skills}\n"
    f"Recent Job Titles: {recent_jobs}\n"
    f"Suggest suitable future job titles and how to prepare for them."
)


# ### 5-Call the GPT Model for Each Template

# In[99]:


# Send Template 1 to GPT
response_1 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt_1}
    ]
)

# Send Template 2 to GPT
response_2 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt_2}
    ]
)

# Print the outputs
print("🔹 Template 1 (Simple Prompt) Output:\n")
print(response_1.choices[0].message.content)

print("\n\n🔹 Template 2 (Role-Based Prompt) Output:\n")
print(response_2.choices[0].message.content)


# ## Comparison of Template Outputs
# 
# ### 1. Depth and Detail
# #### -Template 1 (Simple Prompt):
# Providing a summary of possibale career paths, based on the user’s skills. It briefly discusses some common roles like a Food Microbiologist, Product Development Scientist, and a Sales/Marketing Coordinator. This is informative, but it is rather just a descriptively listed content and does not provide in-depth insights on specific qualification and preparation required for each role.
# #### -Template 2 (Role-Based Prompt): 
# Goes one step further by not only suggesting job titles but even specific preparation steps, such as getting certified (e.g., ISO 9001, Six Sigma, HACCP) or graduate studies. It shows a better understanding of how the user can grow into each position, so the advice is more specific and actionable.
# 
# ### 2. Tone and Professionalism
# #### -Template 1: 
# Uses a helpful tone  but lacks the authoritative voice that would be expected from a professional advisor. It sounds more like a "here are some suggestions" list
# #### -Template 2:
# Adopts the voice of a career advisor, as intended. It recommends targeted actions, like getting certifications and degrees, which reflects a more expert and structured tone. This approach makes the AI sound more trustworthy and intentional in guiding the user’s career.
# 
# ### 3. Structure and Readability
# #### -Template 1:
# Is narrative-based, starting with an overview of the user’s qualifications and then listing potential careers. This makes it easy to read, but not highly structured for planning.
# #### -Template 2: 
# Is role-by-role structured, where each potential career path is clearly laid out with a job title and its associated preparation steps. This format is more useful for users who want to take concrete steps toward a specific role.
# 
# ### 4.Use Case Suitability
# #### -Template 1:
# Is better for quick inspiration or general guidance, especially for users just exploring career possibilities.
# #### -Template 2: 
# Is more appropriate for users who are serious about career planning and need focused advice. It fits the purpose of a professional AI-enhanced advisor much better.
# 
# 
# 
# -------------------------------------------------------------------------------------------
# Finally, Template 2 (Role-Based Prompt) offers better value for system. Therefore, Template 2 is the better fit for integration.

# ## Comparison of Template Outputs
# 
# ### 1. Depth and Detail 
# #### -
# Providing a summary of possibale career paths, based on the user’s skills. It briefly discusses some common roles like a Food Microbiologist, Product Development Scientist, and a Sales/Marketing Coordinator. This is informative, but it is rather just a descriptively listed content and does not provide in-depth insights on specific qualification and preparation required for each role.
# #### -Template 2 (Role-Based Prompt): 
# Goes one step further by not only suggesting job titles but even specific preparation steps, such as getting certified (e.g., ISO 9001, Six Sigma, HACCP) or graduate studies. It shows a better understanding of how the user can grow into each position, so the advice is more specific and actionable.
# 
# ### 2. Tone and Professionalism
# #### -Template 1: 
# Uses a helpful tone  but lacks the authoritative voice that would be expected from a professional advisor. It sounds more like a "here are some suggestions" list
# #### -Template 2:
# Adopts the voice of a career advisor, as intended. It recommends targeted actions, like getting certifications and degrees, which reflects a more expert and structured tone. This approach makes the AI sound more trustworthy and intentional in guiding the user’s career.
# 
# ### 3. Structure and Readability
# #### -Template 1:
# Is narrative-based, starting with an overview of the user’s qualifications and then listing potential careers. This makes it easy to read, but not highly structured for planning.
# #### -Template 2: 
# Is role-by-role structured, where each potential career path is clearly laid out with a job title and its associated preparation steps. This format is more useful for users who want to take concrete steps toward a specific role.
# 
# ### 4.Use Case Suitability
# #### -Template 1:
# Is better for quick inspiration or general guidance, especially for users just exploring career possibilities.
# #### -Template 2: 
# Is more appropriate for users who are serious about career planning and need focused advice. It fits the purpose of a professional AI-enhanced advisor much better.
# 
# 
# 
# -------------------------------------------------------------------------------------------
# Finally, Template 2 (Role-Based Prompt) offers better value for system. Therefore, Template 2 is the better fit for integration.

# ## Justification for the Chosen Template
# After comparing the outputs of both templates the Role-Based Prompt is the most suitable template for integration. This template allows the Generative AI model to simulate the behavior of a professional career advisor by considering multiple aspects of the user's background, including their skills, courses, and recent job titles. Unlike the Simple Instruction Prompt which offers general suggestions based on limited input, the Role-Based Prompt encourages the model to generate more personalized and structured recommendation. This aligns directly with our system’s goal of providing users with realistic and helpful career recommendation. By framing the prompt in a way that gives the model a defined role and rich context, the quality and relevance of the output are significantly enhanced, making it the preferred template for integration.
