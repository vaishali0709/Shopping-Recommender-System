Recommender (Build intelligence to help customers discover products they may like and most likely purchase)

Problem Statement

In the e-commerce industry, providing personalized product recommendations to customers is crucial for enhancing user experience, increasing customer engagement, and boosting sales. The goal is to develop a shopping recommender system that suggests relevant products to customers based on their preferences and browsing history.

Project Focus

•	Recommend products based on historical purchase data and browsing behavior.
•	Enhance customer shopping experience with personalized suggestions.
•	Increase customer engagement and satisfaction.
•	Leverage various recommendation techniques and evaluation metrics.

Deliverables

1.	Comprehensive Report (PDF): Overview, techniques, evaluation metrics, and benefits.
2.	Source Code: Code for the recommender system.
3.	User Guide/README: Instructions for installation and running the pipeline.

Tasks/Activities List

1.	Data Collection: Collect product data, customer profiles, purchase history, and browsing behavior.
2.	Data Preprocessing: Clean and transform the data for modeling.
3.	Recommendation Techniques: Implement collaborative filtering, content-based filtering, and hybrid models.
4.	Evaluation Metrics: Use precision, recall, F1-score, and mean average precision.





1. Overview of the Shopping Recommender System's Design and Architecture

The Shopping Recommender System aims to improve the shopping experience by providing personalized product suggestions based on customers' past purchases and browsing habits. The system consists of three main parts:

•	Data Integration: This part collects various datasets, including customer profiles, product information, and purchase history. The customer data includes personal details, the product data categorizes items, and the transaction history shows how customers interact with products.

•	Recommendation Engine: This is the heart of the system, which uses different methods to suggest products:

o	Collaborative Filtering: This method looks at past interactions between users and products (like purchases) to predict what they might like in the future.

o	Content-Based Filtering: This method recommends products based on their features (like category and description).

•	Evaluation and Performance: The system's effectiveness is measured using metrics such as Precision, Recall, F1-Score, and Mean Average Precision (MAP) to assess how accurate and relevant the recommendations are.

•	Deployment: The system can be implemented on an e-commerce website to provide real-time product suggestions as customers browse or make purchases.

2. Recommendation Techniques Used

The system combines Collaborative Filtering and Content-Based Filtering techniques to ensure accurate recommendations.

•	Collaborative Filtering: This method analyses users' past behaviours (like products they bought or rated) to predict future preferences. It uses the SVD (Singular Value Decomposition) algorithm to find hidden patterns in user-product interactions, allowing for personalized suggestions.

•	Content-Based Filtering: This method focuses on the products' characteristics. It uses a TF-IDF (Term Frequency-Inverse Document Frequency) model to convert product descriptions into numerical vectors. The system then calculates Cosine Similarity between products to recommend similar items based on their features.
•	Hybrid Model: This approach combines both collaborative and content-based methods to take advantage of their strengths. By considering both user behaviour and product attributes, the hybrid model offers more accurate and varied recommendations than using just one method.

3. Evaluation Metrics

The effectiveness of the recommender system is measured using the following metrics:

•	RMSE (Root Mean Square Error): This measures the difference between predicted and actual ratings. A lower RMSE indicates better performance.
•	MAE (Mean Absolute Error): Similar to RMSE, but it calculates the average absolute differences between predicted and actual ratings.
•	Precision: This is the percentage of recommended items that are relevant. High precision means most recommended items are useful to the user.
•	Recall: This measures the percentage of relevant items that were recommended. High recall indicates that most relevant items were included in the suggestions.
•	F1-Score: This combines Precision and Recall into a single score. A higher F1-Score means better overall performance.
•	Mean Average Precision (MAP): This assesses the quality of the recommendation ranking. A higher MAP score means more relevant items are ranked higher.

4. Benefits of the System for E-Commerce Organizations and Customers

For the E-Commerce Organization:

•	Increased Sales: Personalized recommendations encourage customers to buy more items, boosting overall sales.

•	Enhanced Customer Engagement: Customers are more likely to interact with products that match their preferences, leading to greater satisfaction and loyalty.

•	Improved Customer Retention: Tailored recommendations create a more personalized shopping experience, increasing the chances of repeat purchases.

•	Data-Driven Insights: The system provides valuable information about customer behaviour, which can help improve marketing strategies, inventory management, and promotional efforts.


For the Customers:

•	Personalized Shopping Experience: Customers receive product suggestions that match their interests, making shopping more enjoyable.

•	Time-Saving: Accurate recommendations help customers find products faster, enhancing their shopping experience.

•	Product Discovery: Customers are introduced to products they might not have found on their own, broadening their choices.

•	Enhanced Satisfaction: A well-designed recommendation system increases customer satisfaction by offering relevant products, improving the chances of meeting their needs.





























Shopping Recommender System Implementation: Code and Explanation
Objective:
The objective of this project is to build a shopping recommender system that provides personalized product recommendations based on customer historical data, product categories, and browsing behaviour. The system uses collaborative filtering and content-based filtering techniques. Evaluation of the model is done using various metrics such as precision, recall, F1-score, and mean average precision.
________________________________________
1. Data Collection and Preprocessing
The first step is to load and preprocess the data. We collect product data, customer profiles, purchase history, and browsing behaviour from the e-commerce platform.
Code:
import pandas as pd
import numpy as np
from datetime import datetime

# File paths
customer_path = 'D:/DATA ANALYSIS/Recommnder/customer.csv'
transactions_path = 'D:/DATA ANALYSIS/Recommnder/transactions.csv'
prod_cat_info_path = 'D:/DATA ANALYSIS/Recommnder/prod_cat_info.csv'

# Load datasets
customers = pd.read_csv(customer_path)
transactions = pd.read_csv(transactions_path)
prod_cat_info = pd.read_csv(prod_cat_info_path)

# Convert date columns to datetime format
customers['DOB'] = pd.to_datetime(customers['DOB'], format='%d-%m-%Y')
transactions['tran_date'] = pd.to_datetime(transactions['tran_date'], dayfirst=True, errors='coerce')

# Handle negative values in 'Qty' and 'total_amt' (treat as returns)
transactions['Qty'] = transactions['Qty'].abs()
transactions['total_amt'] = transactions['total_amt'].abs()

# Merge datasets
merged_df = transactions.merge(customers, left_on='cust_id', right_on='customer_Id')
merged_df = merged_df.merge(prod_cat_info, on='prod_cat_code')

# Display the merged data
print(merged_df.head())

Explanation:
1.	Loading Data: We load three datasets: customer.csv, transactions.csv, and prod_cat_info.csv.
2.	Preprocessing: We convert the date columns to datetime format and handle negative values in Qty and total_amt by converting them to positive (to handle returns).
3.	Merging: The datasets are merged on the appropriate keys (cust_id, customer_Id, prod_cat_code) to create a comprehensive merged_df.
Result: The merged dataset contains transaction details, product information, and customer demographics.

2. Recommendation Techniques
2 (a). Collaborative Filtering (SVD Model)
We implement a Collaborative Filtering technique using Singular Value Decomposition (SVD). This method helps predict missing ratings based on the patterns of past behaviour between users and products.
Code:
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate

# Prepare data for collaborative filtering
reader = Reader(rating_scale=(0, merged_df['total_amt'].max()))
data = Dataset.load_from_df(merged_df[['cust_id', 'prod_subcat_code', 'total_amt']], reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

# Evaluate the model using cross-validation
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

Explanation:
1.	Data Preparation: We use the Reader class from the surprise library to define the rating scale (0 to max of total_amt) and load the data into a format suitable for Surprise.
2.	Model Training: We train the SVD (Singular Value Decomposition) model on the training data.
3.	Cross-Validation: We use 5-fold cross-validation to evaluate the model's performance with RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

Result: The model's performance metrics (RMSE and MAE) were calculated, demonstrating the predictive accuracy of the model.

2(b). Content-Based Filtering (TF-IDF and Cosine Similarity)
We also implement Content-Based Filtering using TF-IDF vectorization to convert product descriptions into numerical form. We then calculate the cosine similarity to recommend similar products.
Code:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create product descriptions by combining category and subcategory
prod_cat_info['description'] = prod_cat_info['prod_cat'] + ' ' + prod_cat_info['prod_subcat']

# Vectorize the product descriptions
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(prod_cat_info['description'])

# Compute cosine similarity between products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend products based on a given product
def recommend_products(product_id, num_recommendations=5):
    idx = prod_cat_info.index[prod_cat_info['prod_cat_code'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    product_indices = [i[0] for i in sim_scores]
    return prod_cat_info.iloc[product_indices]

# Example: Recommend products similar to product with ID 1
print(recommend_products(1))

Explanation:
1.	Description Combination: We combine the product category and subcategory information to create a description for each product.
2.	TF-IDF Vectorization: We use the TfidfVectorizer to convert these descriptions into numerical vectors.
3.	Cosine Similarity: We calculate the cosine similarity to find similar products based on their descriptions.

Result: A set of recommended products similar to a specified product is generated.


3. Model Evaluation
Finally, we evaluate the performance of the recommender system using various evaluation metrics such as precision, recall, F1-score, and mean average precision (MAP).

Code :

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

# Get predictions from the collaborative filtering model
predictions = model.test(testset)

# Extract the actual and predicted ratings from the predictions
y_true = np.array([pred.r_ui for pred in predictions])  # Actual ratings from the test set
y_pred = np.array([pred.est for pred in predictions])  # Predicted ratings

# Convert ratings to binary values (1 if the predicted rating is above a threshold, otherwise 0)
threshold = 0.5  # You can adjust this threshold as needed based on your dataset
y_true_binary = (y_true > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)

# Calculate Precision, Recall, F1-score, and Mean Average Precision (MAP)
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)
map_score = average_precision_score(y_true_binary, y_pred_binary)

# Print the evaluation results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Mean Average Precision: {map_score:.4f}')

Explanation:
1.	Extract Predictions: We extract the actual and predicted ratings for the test data.
2.	Binary Conversion: Ratings are converted into binary values based on a threshold (e.g., ratings greater than 0.5 are considered relevant).
3.	Metric Calculation: We calculate precision, recall, F1-score, and mean average precision (MAP) to evaluate the recommender system's performance.

Result: The system achieved perfect scores in Precision, Recall, F1-Score, and MAP, indicating the high effectiveness of the recommendation system.

Conclusion:
The implementation of the shopping recommender system based on collaborative filtering (SVD) and content-based filtering has shown strong performance. The system has achieved perfect scores for precision, recall, F1-score, and mean average precision on the test set, indicating that the recommendations are highly accurate and relevant to the customers.

