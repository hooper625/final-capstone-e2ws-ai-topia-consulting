import sys
from pathlib import Path
# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Sklearn - preprocessing
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA

# Sklearn - models
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Sklearn - models selection and evaluation
from sklearn.model_selection import LearningCurveDisplay

# Sklearn - evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,  RocCurveDisplay, PrecisionRecallDisplay
)

from imblearn.over_sampling import SMOTE

from pipelines.data_pipeline import plot_feature_importance, print_model_report, plot_prediction_probabilities

# Settings
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


# ====================================================================================
# classification metrics for Train and Test
# ====================================================================================
def evaluate_classification_model(trained_model, X_train, X_test, y_train, y_test, model_name):
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

# =============================================================================
# This helper function balances the training data using over-sampling (SMOTE)
# =============================================================================
def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE to balance the target classes in the training set.

    Returns:
        Resampled X_train and y_train
    """
    # Initialize SMOTE
    smote = SMOTE(random_state=42)
    
    # Fit and resample only the training data to avoid data leakage
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Display the new class distribution
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"Resampled class distribution: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled

# =============================================================================
# This helper function performs Grid Search to find optimal model hyperparameters
# =============================================================================
def tune_classifier(trained_model, param_grid, X_train, y_train, cv=5):
    """Run GridSearchCV to optimize model parameters based on F1-score.

    Args:
        model: The estimator to tune
        param_grid: Dictionary of parameters to test
        X_train, y_train: Training data
        cv: Number of cross-validation folds

    Returns:
        The best estimator found by the grid search
    """
    # Initialize Grid Search with weighted F1 to handle potential class imbalance
    grid_search = GridSearchCV(
        estimator=trained_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1, # Use all available CPU cores
        verbose=1
    )

    # Fit to the training data
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# =============================================================================
# This helper function engineers new features to improve model predictive power
# =============================================================================
def engineer_classification_features(df):
    """Create new features from existing columns.

    Examples:
    - Parse datetime columns -> hour, day_of_week, month
    - Create binary flags from categorical data
    - Bin continuous variables into categories
    - Interaction features

    Returns:
        DataFrame with new feature columns
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Temporal Features: Extracting components from datetime objects
    # Assuming 'timestamp' column exists; change name as needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 2. Binning: Converting continuous variables into discrete categories
    # Useful for non-linear relationships in models like Logistic Regression
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=[0, 1, 2, 3])

    # 3. Interaction Features: Combining two features to capture synergy
    # Example: Multiplying two numeric scores
    if 'feature_a' in df.columns and 'feature_b' in df.columns:
        df['a_b_interaction'] = df['feature_a'] * df['feature_b']

    return df


# =============================================================================
# This helper function Prediction Probability Histograms for each class
# =============================================================================
def plot_advanced_evaluation(trained_model, X_test, y_test, model_name):
    """
    Generates ROC and Precision-Recall plots for the trained_model.
    Best for balanced datasets. 
    It shows the trade-off between sensitivity and specificity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. ROC Curve
    RocCurveDisplay.from_estimator(trained_model, X_test, y_test, ax=ax1)
    ax1.set_title(f"ROC Curve - {model_name}")

    # 2. Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(trained_model, X_test, y_test, ax=ax2)
    ax2.set_title(f"Precision-Recall - {model_name}")

    plt.tight_layout()
    plt.show()

# =============================================================================
# This helper function Prediction Probability Histograms for each class
# =============================================================================
def plot_learning_curve(trained_model, X, y, model_name):
    """
    Generates a Learning Curve to check for Overfitting.
    A Learning Curve plots the training and validation scores against the number of training samples.
     It helps to visualize how the model's performance evolves as it sees more data.
    If the training score is much higher than the validation score, it indicates overfitting.
    If both scores are low, it indicates underfitting.
    If both scores are close and high, it indicates good generalization."""
    print(f"Generating Learning Curve for {model_name}...")
    display = LearningCurveDisplay.from_estimator(
       trained_model, X, y, score_name="Accuracy", train_sizes=np.linspace(0.1, 1.0, 5)
    )
    display.ax_.set_title(f"Learning Curve: {model_name}")
    plt.show()

# =============================================================================
# This helper function reduces high-dimensional data to 2D for visualization
# =============================================================================
def plot_pca_2d(X, y, title="PCA 2D Projection"):
    """Reduces features to 2 components and plots class clusters."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 7))
    # Standardize y to handle both numeric and categorical targets
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    print(f"Explained variance ratio (first 2 components): {pca.explained_variance_ratio_.sum():.2%}")


    
# ======================================================================================================
# Hist Gradient Boosting Classification Component
# ======================================================================================================
def run_hist_gradient_boosting(X_train, X_test, y_train, y_test, **kwargs):
    """
    Trains a HistGradientBoostingClassifier and plots permutation importance.
    """
    
    # Initialize with dynamic parameters
    params = {'max_iter': 100, 'max_depth': 5, 'random_state': 42}
    params.update(kwargs)
    trained_model =HistGradientBoostingClassifier(**params)

    # Evaluate using your engine and get predictions for the test set
    results, trained_model, y_test_pred = evaluate_classification_model(
        trained_model, X_train, X_test, y_train, y_test, "Hist Gradient"
    )

    #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "Hist Gradient")

    #Plot Importance 
    plot_feature_importance(trained_model, X_test, y_test, "Hist Gradient")

    # Advanced Visuals
    # Show if the classes are separable (only needs to be run once per dataset, really)
    plot_pca_2d(X_test, y_test, title="Hist Gradient - Data Separation (PCA)")
    
    # Show if the model is confident
    plot_prediction_probabilities(trained_model, X_test, "Hist Gradient")
    
    return results, trained_model

# =============================================================================
# This helper function trains a RandomForest with dynamic parameters
# =============================================================================
def run_random_forest(X_train, X_test, y_train, y_test, **kwargs):
    """Trains a RandomForestClassifier with dynamic control over hyperparameters."""
    params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    params.update(kwargs)
    
    trained_model =RandomForestClassifier(**params)
    results, trained_model, y_test_pred = evaluate_classification_model(trained_model, X_train, X_test, y_train, y_test, "Random Forest")

       #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "Random Forest")

    #Plot Importance 
    plot_feature_importance(trained_model, X_test, y_test, "Random Forest")

    # Advanced Visuals
    # Show if the classes are separable (only needs to be run once per dataset, really)
    plot_pca_2d(X_test, y_test, title="Random Forest - Data Separation (PCA)")
    
    # Show if the model is confident
    plot_prediction_probabilities(trained_model, X_test, "Random Forest")
    return results, trained_model

# ======================================================================================================
# Decision Tree Classification Component
# ======================================================================================================
def run_decision_tree(X_train, X_test, y_train, y_test, max_depth=5):
    """Trains a DecisionTreeClassifier and returns results and trained_model."""
    trained_model =DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # Standardized evaluation call
    results, trained_model, y_test_pred = evaluate_classification_model(
       trained_model, X_train, X_test, y_train, y_test, "Decision Tree"
    )

    #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "Decision Tree")

    #Plot Importance 
    plot_feature_importance(trained_model, X_test, y_test, "Decision Tree")
    
    # Advanced Visuals
    # Show if the classes are separable (only needs to be run once per dataset, really)
    plot_pca_2d(X_test, y_test, title="Decision Tree - Data Separation (PCA)")
    
    # Show if the model is confident
    plot_prediction_probabilities(trained_model, X_test, "Decision Tree")

    return results, trained_model

# =============================================================================
# Dynamic XGBoost Component with Feature Importance
# =============================================================================
def run_xgb_classifier_feature(X_train, X_test, y_train, y_test, **kwargs):
    """Trains XGBoost and automatically triggers importance plotting."""
    params = {
        'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
        'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42,
        'objective': 'multi:softprob'
    }
    params.update(kwargs)

    trained_model =XGBClassifier(**params)
    
    # 1. Evaluate
    results, trained_model, y_test_pred= evaluate_classification_model(
       trained_model, X_train, X_test, y_train, y_test, "XGBoost"
    )

    #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "XGBoost")

    #Plot Importance 
    plot_feature_importance(trained_model, X_test, y_test, "XGBoost")

    return results, trained_model

# ======================================================================================================
# Gradient Boosting (Standard) Component
# ======================================================================================================
def run_gradient_boosting(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=5):
    """Trains a GradientBoostingClassifier and returns results and trained_model."""
    trained_model =GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    results, trained_model, y_test_pred = evaluate_classification_model(
       trained_model, X_train, X_test, y_train, y_test, "Gradient Boosting"
    )

    #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "Gradient Boosting")

    #Plot Importance 
    plot_feature_importance(trained_model, X_test, y_test, "Gradient Boosting")

# Advanced Visuals
    # Show if the classes are separable (only needs to be run once per dataset, really)
    plot_pca_2d(X_test, y_test, title="Gradient Boosting - Data Separation (PCA)")
    
    # Show if the model is confident
    plot_prediction_probabilities(trained_model, X_test, "Gradient Boosting")

    return results, trained_model

# =============================================================================
# This helper function trains a KNN model with dynamic parameters
# =============================================================================
def run_knn(X_train, X_test, y_train, y_test, **kwargs):
    """Trains a KNeighborsClassifier with dynamic control (e.g., n_neighbors)."""
    params = {'n_neighbors': 5}
    params.update(kwargs)
    
    trained_model =KNeighborsClassifier(**params)
    results, trained_model, y_test_pred = evaluate_classification_model(trained_model, X_train, X_test, y_train, y_test, "KNN")
    
    #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "KNN")

    # #Plot Importance 
    # plot_feature_importance(trained_model, X_test, y_test, "KNN")

    # # Advanced Visuals
    # # Show if the classes are separable (only needs to be run once per dataset, really)
    # plot_pca_2d(X_test, y_test, title="KNN - Data Separation (PCA)")
    
    # # Show if the model is confident
    # plot_prediction_probabilities(trained_model, X_test, "KNN")

    return results, trained_model

# ======================================================================================================
# SVM Classification Component
# ======================================================================================================
def run_svm_linear(X_train, X_test, y_train, y_test, C=1.0):
    """Trains an SVM with a linear kernel and returns results and trained_model."""
    trained_model = SVC(kernel='linear', C=C, gamma='scale', probability=True, random_state=42)
    
    results, trained_model, y_test_pred = evaluate_classification_model(
       trained_model, X_train, X_test, y_train, y_test, "SVM linear"
    )
    
    #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "SVM linear")

    #Plot Importance 
    plot_feature_importance(trained_model, X_test, y_test, "SVM linear")

        # Advanced Visuals
    # Show if the classes are separable (only needs to be run once per dataset, really)
    plot_pca_2d(X_test, y_test, title="SVM linear - Data Separation (PCA)")
    
    # Show if the model is confident
    plot_prediction_probabilities(trained_model, X_test, "SVM linear")

    return results, trained_model

# ======================================================================================================
# Voting Classifier Component
# ======================================================================================================
def run_voting_classifier(X_train, X_test, y_train, y_test, voting='soft', estimators_list=None):
    """
    Trains a VotingClassifier. 
    Expects estimators_list in format: [('name', model_object), ...]
    """
    trained_model =VotingClassifier(estimators=estimators_list, voting=voting)
    
    results, trained_model, y_test_pred = evaluate_classification_model(
       trained_model, X_train, X_test, y_train, y_test, "Voting Classifier"
    )
    
    #Output Detailed Metrics (The part you requested)
    print_model_report(y_test, y_test_pred, "Voting Classifier")

    #Plot Importance 
    plot_feature_importance(trained_model, X_test, y_test, "Voting Classifier")

        # Advanced Visuals
    # Show if the classes are separable (only needs to be run once per dataset, really)
    plot_pca_2d(X_test, y_test, title="Voting Classifier - Data Separation (PCA)")
    
    # Show if the model is confident
    plot_prediction_probabilities(trained_model, X_test, "Voting Classifier")

    return results, trained_model

