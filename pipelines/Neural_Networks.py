import sys
from pathlib import Path
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Neural Networks
from sklearn.neural_network import MLPClassifier

# Sklearn - evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, classification_report, confusion_matrix,  RocCurveDisplay, PrecisionRecallDisplay
)
from data_pipeline import plot_feature_importance, print_model_report, plot_prediction_probabilities
from sklearn.feature_extraction.text import TfidfVectorizer

# Settings
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# ========================================================================================
# Neural Network evaluation function - similar to your existing one but tailored for NN
# ========================================================================================
def evaluate_neural_network_model(trained_model, X_train, X_test, y_train, y_test, model_name):
    """Train model and return classification metrics for Train and Test."""
    # Train the model
    trained_model.fit(X_train, y_train)
    
    # Get both sets of predictions
    y_train_pred = trained_model.predict(X_train)
    y_test_pred = trained_model.predict(X_test)
    
    # Calculate metrics for both sets
    results = {
        'Model': model_name,
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Train F1 (weighted)': f1_score(y_train, y_train_pred, average='weighted'),
        'Test F1 (weighted)': f1_score(y_test, y_test_pred, average='weighted'),
        'Train Precision': precision_score(y_train, y_train_pred, average='weighted'),
        'Test Precision': precision_score(y_test, y_test_pred, average='weighted')
    }
    
    return results,trained_model, y_test_pred

from sklearn.neural_network import MLPClassifier

# ========================================================================================
# Trains a Feedforward Neural Network (MLP).
# ========================================================================================
def run_mlp_classifier(X_train, X_test, y_train, y_test, **kwargs):
    """
    Trains a Feedforward Neural Network (MLP).
    Best for: Tabular data (Salary Predictor, etc.)
    Recommended Data: Scaled
    """
    # Default parameters that can be overridden by kwargs
    params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 500,
        'random_state': 42
    }
    params.update(kwargs)
    
    trained_model = MLPClassifier(**params)
    
    # Standardized evaluation and reporting
    results, trained_model, y_test_pred = evaluate_neural_network_model(
        trained_model, X_train, X_test, y_train, y_test, "MLP Neural Net"
    )
    
    print_model_report(y_test, y_test_pred, "MLP Neural Net")
    plot_feature_importance(trained_model, X_test, y_test, "MLP Neural Net")
    plot_prediction_probabilities(trained_model, X_test, "MLP Neural Net")
    
    return results, trained_model

# ========================================================================================
# Trains a Feedforward Neural Network (MLP).
# ========================================================================================
def extract_text_features(df, text_col, max_features=1000):
    """
    Converts cleaned text into a TF-IDF matrix.
    Higher scores = more unique/important words.
    """
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df[text_col])
    
    # Convert to a DataFrame to join back to your main features
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    return tfidf_df, tfidf