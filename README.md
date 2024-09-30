Overview
This project implements a data classification and clustering algorithm using Logistic Regression, Bayesian Ridge regression, and k-means-like clustering. The main objective is to identify unknown data points from two datasets by:

Standardizing the data,
Classifying new data using pre-trained models,
Clustering unknown data, and
Calculating accuracy using custom-defined centroids.
The workflow includes loading the data, preprocessing it, training classifiers, and performing unsupervised clustering on unknown data.

Files
train_data.csv: The training dataset used for building models.
train_label.csv: Labels corresponding to the training data.
test_data.csv: The test dataset for evaluating the model.
test_label.csv: Labels corresponding to the test data.
Requirements
This project uses Python 3.x and requires the following libraries:

numpy
pandas
scikit-learn
Install the required dependencies using the command:

Functions
similar(*val, threshold=0.1)
This function checks if the maximum difference between given values is below a specified threshold, determining similarity between probability predictions.

preprocess(data)
Standardizes the given data using StandardScaler from sklearn. This is crucial for ensuring all data is on the same scale for classification and clustering tasks.

classify(old_label, old_data, new_label, new_data, mode=0)
Mode 0: Performs Logistic Regression to classify new data. It flags unknown data points based on a threshold.
Mode 1: Uses Bayesian Ridge Regression for classification and flags unknown data points.
Returns the indices of unknown data points.
gen_old_centroid(data, k, old_label, mode=0)
Generates centroids for clustering by averaging the positions of labeled data. It assigns labels based on the given mode (classification mode).

cluster(data, k, max_iter, centroids=None, mode=0)
Performs clustering using k-means-like algorithm:

Randomly selects initial centroids.
Refines the centroids iteratively based on the nearest points.
Optionally uses predefined centroids for clustering unknown data.
gen_rate(label, targetlabel)
Calculates the accuracy of clustering by comparing predicted labels with actual labels. It generates permutations of possible labels and selects the best match.

test()
Main function that:

Classifies new data using Logistic Regression.
Identifies and clusters unknown data.
Combines old and new centroids and performs a final clustering of the entire dataset.
Returns the highest accuracy from the final clustering.
Main Execution
The script runs the test() function multiple times and prints the final probability accuracy for classifying unknown test data.

Usage
Prepare your data: Ensure that the CSV files (train_data.csv, train_label.csv, test_data.csv, test_label.csv) are in the correct format and in the same directory as the script.

Run the script:

python your_script.py
Output:

The script will output the value counts of the labels in the test set and display the accuracy of the classification and clustering process.
Example Output

KIRC    0.30
BRCA    0.40
LUAD    0.30
Probability= [0.85, 0.83, 0.82]
