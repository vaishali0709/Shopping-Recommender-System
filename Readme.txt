User Guide for Shopping Recommender System
Overview

This Shopping Recommender System is designed to provide personalized product recommendations to customers of an e-commerce platform. By using Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF + Cosine Similarity), the system helps customers discover products they are likely to be interested in based on their purchase history and product descriptions.

This guide provides instructions on how to install and run the end-to-end pipeline, including data preparation, model training, and evaluation.
Table of Contents

    System Requirements
    Installation Instructions
    Running the Pipeline
    Data Preparation
    Model Training
    Model Evaluation
    Troubleshooting
    License

System Requirements

Before running the Shopping Recommender System, ensure that the following software is installed on your machine:

    Python 3.x: Ensure Python 3.7 or above is installed.
    Pip: Package manager for installing Python dependencies.

Required Python Libraries

    pandas: Data manipulation and cleaning
    numpy: Numerical operations
    scikit-learn: Machine learning algorithms and evaluation metrics
    surprise: Collaborative filtering models
    sklearn: For vectorization and similarity computation
    cosine_similarity: To compute product similarities

These dependencies can be installed using the requirements.txt file provided.
Installation Instructions
Step 1: Install Dependencies

Use pip to install the required Python libraries. Create a virtual environment (optional but recommended) and run the following command:

pip install -r requirements.txt

This will install all the necessary dependencies.

Running the Pipeline
Step 1: Prepare the Data

Ensure the following CSV files are available in the correct directory (D:/DATA ANALYSIS/Recommnder/):

    customer.csv: Contains customer data such as customer_Id, DOB, and gender.
    transactions.csv: Contains transaction data like cust_id, prod_cat_code, Qty, total_amt, and tran_date.
    prod_cat_info.csv: Contains product category data such as prod_cat_code, prod_cat, prod_subcat_code, and prod_subcat.

Once the datasets are in place, the following Python script loads and cleans the data:

import pandas as pd

# Load datasets
customers = pd.read_csv('D:/DATA ANALYSIS/Recommnder/customer.csv')
transactions = pd.read_csv('D:/DATA ANALYSIS/Recommnder/transactions.csv')
prod_cat_info = pd.read_csv('D:/DATA ANALYSIS/Recommnder/prod_cat_info.csv')

# Data cleaning and merging
customers['DOB'] = pd.to_datetime(customers['DOB'], format='%d-%m-%Y')
transactions['tran_date'] = pd.to_datetime(transactions['tran_date'], dayfirst=True, errors='coerce')
transactions['Qty'] = transactions['Qty'].abs()
transactions['total_amt'] = transactions['total_amt'].abs()

# Merge datasets
merged_df = transactions.merge(customers, left_on='cust_id', right_on='customer_Id')
merged_df = merged_df.merge(prod_cat_info, on='prod_cat_code')

Step 2: Train the Model

After preparing the data, train the collaborative filtering and content-based models. Below is the code to train the SVD model using the surprise library and the Content-Based Filtering model using TF-IDF and Cosine Similarity:
Collaborative Filtering (SVD)

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Prepare data for collaborative filtering
reader = Reader(rating_scale=(0, merged_df['total_amt'].max()))
data = Dataset.load_from_df(merged_df[['cust_id', 'prod_subcat_code', 'total_amt']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

Content-Based Filtering (TF-IDF + Cosine Similarity)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create product descriptions by combining category and subcategory
prod_cat_info['description'] = prod_cat_info['prod_cat'] + ' ' + prod_cat_info['prod_subcat']

# Vectorize descriptions using TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(prod_cat_info['description'])

# Calculate cosine similarity between products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

Step 3: Evaluate the Model

After training, evaluate the performance of the collaborative filtering model using Precision, Recall, F1-Score, and Mean Average Precision (MAP).

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

# Get predictions from the collaborative filtering model
predictions = model.test(testset)

# Extract actual and predicted ratings
y_true = np.array([pred.r_ui for pred in predictions])
y_pred = np.array([pred.est for pred in predictions])

# Convert to binary values (1 if predicted rating > threshold, otherwise 0)
threshold = 0.5
y_true_binary = (y_true > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)

# Calculate metrics
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)
map_score = average_precision_score(y_true_binary, y_pred_binary)

# Print the evaluation results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Mean Average Precision: {map_score:.4f}')

Step 4: Make Recommendations

Use the collaborative filtering or content-based models to generate recommendations for users based on their preferences. For content-based recommendations, the code below recommends similar products:

# Function to recommend products
def recommend_products(product_id, num_recommendations=5):
    idx = prod_cat_info.index[prod_cat_info['prod_cat_code'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    product_indices = [i[0] for i in sim_scores]
    return prod_cat_info.iloc[product_indices]

# Example: Recommend products similar to product with ID 1
print(recommend_products(1))

Troubleshooting

    Missing Data: Ensure that all required datasets are present in the specified directory (D:/DATA ANALYSIS/Recommnder/). Missing files can cause errors during execution.
    Library Installation Issues: If you encounter errors during library installation, ensure that you are using the correct version of Python and that all dependencies are correctly listed in the requirements.txt file.
    Model Accuracy: If the evaluation metrics show poor performance, consider adjusting the rating scale, threshold, or hyperparameters of the SVD model.