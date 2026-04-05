import sys
from pathlib import Path

# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Sklearn - evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sklearn - models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Settings
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


# ==============================================================================================
# This helper function evaluates any model - you'll use it throughout this notebook
# ==============================================================================================
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train model and return regression metrics for Train and Test."""
    # Train the model
    model.fit(X_train, y_train)
    
    # Get both sets of predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for both sets
    results = {
        'Model': model_name,
        'Train R2': r2_score(y_train, y_train_pred),
        'Test R2': r2_score(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    return results, model, y_test_pred

# =============================================================================
# Linear Regression to establish a baseline performance
# =============================================================================
def run_linear_baseline(X_train_scaled, X_test_scaled, y_train, y_test, target_range, target_std):
    """
    Trains and evaluates the baseline Linear Regression model.
    Includes context for RMSE relative to the target's range and variation.
    """
    #Create the model instance
    model_linear = LinearRegression()

    # Use the evaluate_regression_model helper (defined in your previous block)
    baseline_results, baseline_trained, baseline_preds = evaluate_regression_model(
        model_linear,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        "Linear Regression Baseline"
    )

    # 3. Format and Display Results
    print("=" * 50)
    print("BASELINE MODEL: Linear Regression")
    print("=" * 50)
    print(f"Train R²:  {baseline_results['Train R2']:.4f}")
    print(f"Test R²:   {baseline_results['Test R2']:.4f}")
    print(f"Test RMSE: {baseline_results['Test RMSE']:,.2f}")
    print(f"Test MAE:  {baseline_results['Test MAE']:,.2f}")

    # 4. RMSE in Context (Crucial for interpreting 'how bad' an error is)
    print(f"\n--- RMSE in Context ---")
    rmse_pct_range = (baseline_results['Test RMSE'] / target_range) * 100
    rmse_pct_std = (baseline_results['Test RMSE'] / target_std) * 100
    
    print(f"RMSE as % of target range: {rmse_pct_range:.1f}%")
    print(f"RMSE as % of target std:   {rmse_pct_std:.1f}%")

    # Return as a tuple just like your original baseline_model variable
    return baseline_results, baseline_trained, baseline_preds

# ==============================================================================================
# Ridge adds L2 regularization to prevent overfitting by penalizing large coefficients
# ==============================================================================================
def run_ridge_model(X_train_scaled, X_test_scaled, y_train, y_test, alpha=100):
    """
    Trains a Ridge model using only scaled data.
    Evaluates performance and prints a formatted metric table.
    """
    # Initialize and Fit (Using Scaled Data)
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)

    # Get built-in scores (R²)
    r2_ridge_train = ridge_model.score(X_train_scaled, y_train) 
    r2_ridge_test = ridge_model.score(X_test_scaled, y_test)

    #Generate Predictions for Manual Metrics
    y_pred_Ridge = ridge_model.predict(X_test_scaled)

    # Calculate metrics
    rmse_Ridge = np.sqrt(mean_squared_error(y_test, y_pred_Ridge))
    r2_manual = r2_score(y_test, y_pred_Ridge)

    # 5. Print the formatted table
    print("-" * 30)
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    print(f"{'Train R²':<15} | {r2_ridge_train:.4f}")
    print(f"{'Test R²':<15} | {r2_ridge_test:.4f}") 
    print(f"{'R² Gap':<15} | {r2_ridge_train - r2_ridge_test:.4f}")
    print(f"{'Test RMSE':<15} | {rmse_Ridge:.2f}")
    print(f"{'Manual R²':<15} | {r2_manual:.4f}")
    print("-" * 30)

    # 6. Package results for your 'all_results' list
    results = {
        'Model': 'Ridge (Scaled)',
        'Train R2': r2_ridge_train,
        'Test R2': r2_ridge_test,
        'Test RMSE': rmse_Ridge
    }
    
    return results, ridge_model, y_pred_Ridge

# ======================================================================================================
# Lasso adds L1 regularization, which can zero out unimportant features (automatic feature selection)
# ======================================================================================================
def run_lasso_model(X_train_scaled, X_test_scaled, y_train, y_test, all_results_dict, trained_models_dict, alpha=0.1):
    """
    Trains Lasso and updates the existing dictionaries to preserve previous models.
    """
    # 1. Initialize and Fit
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_scaled, y_train)

    # 2. Calculate Metrics
    y_pred = lasso_model.predict(X_test_scaled)
    test_r2 = lasso_model.score(X_test_scaled, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    lasso_results = {
        'Model': f'Lasso (Alpha={alpha})',
        'Test R2': test_r2,
        'Test RMSE': test_rmse
    }

    # 3. Update the Dictionaries (This is where your Baseline/Ridge are preserved!)
    # Check if they are lists or dicts to avoid the index error
    if isinstance(all_results_dict, list):
        all_results_dict.append(lasso_results)
    else:
        all_results_dict['Lasso'] = lasso_results
        
    trained_models_dict['Lasso'] = lasso_model

    # 4. Print Summary
    print(f"Lasso Regression (Alpha={alpha})")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:,.2f}")

    # 5. Feature Selection Logic
    lasso_coefs = pd.Series(lasso_model.coef_, index=X_train_scaled.columns)
    kept_features = lasso_coefs[lasso_coefs != 0]
    
    print(f"\nLasso kept {len(kept_features)} of {len(X_train_scaled.columns)} features.")
    
    return lasso_results, lasso_model, kept_features

# ======================================================================================================
# ElasticNet combines both penalties - it can zero out features AND shrink the rest
# ======================================================================================================
def run_elastic_net_model(X_train_scaled, X_test_scaled, y_train, y_test, all_results_dict, trained_models_dict, alpha=0.001, l1_ratio=0.5):
    """
    Trains ElasticNet (L1 + L2) and updates the global storage.
    """
    # 1. Initialize and Fit
    elastic_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    elastic_model.fit(X_train_scaled, y_train)

    # 2. Calculate Metrics
    r2_train = elastic_model.score(X_train_scaled, y_train)
    r2_test = elastic_model.score(X_test_scaled, y_test)
    y_pred = elastic_model.predict(X_test_scaled)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    elastic_results = {
        'Model': f'ElasticNet (a={alpha}, r={l1_ratio})',
        'Train R2': r2_train,
        'Test R2': r2_test,
        'Test RMSE': rmse_test
    }

    # 3. Internal Storage Update (Dictionary or List safe)
    if isinstance(all_results_dict, list):
        all_results_dict.append(elastic_results)
    else:
        all_results_dict['ElasticNet'] = elastic_results
        
    trained_models_dict['ElasticNet'] = elastic_model

    # 4. Print Summary
    print("ElasticNet (L1 + L2 Combined):")
    print(f"  Train R²: {r2_train:.4f}")
    print(f"  Test R²:  {r2_test:.4f}")
    print(f"  Gap:      {r2_train - r2_test:.4f}")
    print(f"  Test RMSE: {rmse_test:.2f}")

    # 5. Feature Selection Check
    elastic_coefs = pd.Series(elastic_model.coef_, index=X_train_scaled.columns)
    kept_features = elastic_coefs[elastic_coefs != 0]
    print(f"\nElasticNet kept {len(kept_features)} of {len(X_train_scaled.columns)} features.")

    return elastic_results, elastic_model, kept_features

# ======================================================================================================
# PolynomialFeatures` transforms the data to capture curved relationships.
# ======================================================================================================
def run_poly_lasso_model(X_train_scaled, X_test_scaled, y_train, y_test, all_results_dict, trained_models_dict, alpha=0.01, degree=2):
    """
    1. Creates Polynomial Features (Interactions & Squared terms)
    2. Trains a Lasso model to prune the 6,000+ new features
    3. Updates your storage dictionaries automatically
    """
    # --- 1. TRANSFORM ---
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Fit on train, transform both
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Get names for the ~6,500 new columns
    poly_features = poly.get_feature_names_out(X_train_scaled.columns)
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_features, index=X_train_scaled.index)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_features, index=X_test_scaled.index)

    # --- 2. TRAIN ---
    lasso_poly = Lasso(alpha=alpha)
    lasso_poly.fit(X_train_poly_df, y_train)

    # --- 3. EVALUATE ---
    test_r2 = lasso_poly.score(X_test_poly_df, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, lasso_poly.predict(X_test_poly_df)))

    poly_results = {
        'Model': f'Poly Lasso (d={degree}, a={alpha})',
        'Test R2': test_r2,
        'Test RMSE': test_rmse
    }

    # --- 4. STORE ---
    if isinstance(all_results_dict, list):
        all_results_dict.append(poly_results)
    else:
        all_results_dict['PolyLasso'] = poly_results
        
    trained_models_dict['PolyLasso'] = lasso_poly

    # --- 5. REPORT ---
    lasso_coefs = pd.Series(lasso_poly.coef_, index=poly_features)
    kept_features = lasso_coefs[lasso_coefs != 0]
    
    print(f"Polynomial Lasso (Degree {degree}, Alpha {alpha})")
    print(f"Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:,.2f}")
    print(f"Kept {len(kept_features)} out of {len(poly_features)} total features.")
    
    return poly_results, lasso_poly, kept_features

# ======================================================================================================
# Polynomial Ridge can outperform the baseline without the aggressive feature-dropping of Lasso
# ======================================================================================================
def run_poly_ridge_model(X_train_scaled, X_test_scaled, y_train, y_test, all_results_dict, trained_models_dict, alpha=1.0, degree=2):
    """
    1. Generates Polynomial Features (Degree 2)
    2. Trains a Ridge model (L2 Penalty) to handle the 6,000+ features
    3. Updates storage dictionaries automatically
    """
    # --- 1. TRANSFORM ---
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Generate column names for the expanded feature set
    poly_features = poly.get_feature_names_out(X_train_scaled.columns)
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_features, index=X_train_scaled.index)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_features, index=X_test_scaled.index)

    # --- 2. TRAIN & EVALUATE ---
    ridge_poly = Ridge(alpha=alpha)
    ridge_poly.fit(X_train_poly_df, y_train)

    test_r2 = ridge_poly.score(X_test_poly_df, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, ridge_poly.predict(X_test_poly_df)))

    poly_ridge_results = {
        'Model': f'Poly Ridge (d={degree}, a={alpha})',
        'Test R2': test_r2,
        'Test RMSE': test_rmse
    }

    # --- 3. STORE (Safe for lists or dicts) ---
    if isinstance(all_results_dict, list):
        all_results_dict.append(poly_ridge_results)
    else:
        all_results_dict['Polynomial Ridge'] = poly_ridge_results
        
    trained_models_dict['Polynomial Ridge'] = ridge_poly

    # --- 4. REPORT ---
    print(f"Polynomial Ridge (Degree {degree}, Alpha {alpha})")
    print(f"Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:,.2f}")
    
    # Calculate improvement if baseline exists
    if 'Baseline' in all_results_dict:
        # Handling dict vs list indexing for baseline
        base_r2 = all_results_dict['Baseline']['Test R2'] if isinstance(all_results_dict, dict) else all_results_dict[0]['Test R2']
        print(f"Improvement over Baseline: {test_r2 - base_r2:.4f}")
    
    return poly_ridge_results, ridge_poly

# ======================================================================================================
# Decision Trees don't care about "straight lines"—they look for thresholds in your data
# ======================================================================================================
def run_decision_tree_suite(X_train, X_test, y_train, y_test, max_depth=7):
    """
    Trains a Decision Tree and returns the results dict and the model object.
    Does NOT update globals automatically—leaves that to the user.
    """
    # 1. Initialize and Fit
    tree_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree_model.fit(X_train, y_train)
    
    # 2. Calculate Metrics
    y_pred = tree_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 3. Prepare the results dictionary for .append()
    dt_results = {
        'Model': f'Decision Tree (d={max_depth})',
        'Test R2': r2,
        'Test RMSE': rmse
    }

    print(f"Decision Tree (Depth {max_depth}) - R²: {r2:.4f}, RMSE: {rmse:,.2f}")
    
    # Return both so they can be stored in your specific variables
    return dt_results, tree_model

# ======================================================================================================
# Tuned Decision Tree Component
# ======================================================================================================
def run_tuned_tree_model(X_train, X_test, y_train, y_test, param_grid):
    """
    1. Runs GridSearchCV on a Decision Tree with the provided param_grid.
    2. Identifies the best parameters for generalization.
    3. Returns the results dict and the best model object for storage.
    """
    print("Running GridSearchCV... this may take a moment.")
    
    # 1. Initialize the Search
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5, 
        scoring='r2',
        n_jobs=-1  # Uses all your CPU cores to speed it up
    )

    # 2. Fit and Find the Best Model
    grid_search.fit(X_train, y_train)
    best_tree = grid_search.best_estimator_
    
    # 3. Calculate Final Metrics on Test Data
    y_pred = best_tree.predict(X_test)
    tuned_r2 = r2_score(y_test, y_pred)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 4. Prepare Results for .append()
    tuned_dt_results = {
        'Model': 'Tuned Decision Tree',
        'Test R2': tuned_r2,
        'Test RMSE': tuned_rmse,
        'Best Params': grid_search.best_params_
    }

    print(f"Tuned Tree R²: {tuned_r2:.4f}")
    print(f"Best Depth Found: {grid_search.best_params_['max_depth']}")
    
    return tuned_dt_results, best_tree

# ======================================================================================================
# Random Forest is a committee of 100 experts (trees) voting on the result
# ======================================================================================================
def run_random_forest_model(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=7):
    """
    Trains a Random Forest Regressor (Ensemble of Trees).
    Returns the results dict and the fitted model.
    """
    # 1. Initialize (n_jobs=-1 uses all CPU cores for speed)
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42, 
        n_jobs=-1
    )
    
    # 2. Fit and Predict
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # 3. Calculate Metrics
    rf_r2 = r2_score(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    rf_results = {
        'Model': f'Random Forest (n={n_estimators}, d={max_depth})',
        'Test R2': rf_r2,
        'Test RMSE': rf_rmse
    }

    print(f"Random Forest - Test R²: {rf_r2:.4f}, Test RMSE: {rf_rmse:,.2f}")
    
    return rf_results, rf_model

# ======================================================================================================
# Tuned Decision Tree Component
# ======================================================================================================
def run_gradient_boosting_model(X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Trains a Gradient Boosting Regressor (Sequential improvement).
    Returns results dict and fitted model.
    """
    # 1. Initialize
    gbr_model = GradientBoostingRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        random_state=42
    )

    # 2. Train
    gbr_model.fit(X_train, y_train)

    # 3. Predict and Calculate Metrics
    y_pred = gbr_model.predict(X_test)
    gbr_r2 = r2_score(y_test, y_pred)
    gbr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    gbr_results = {
        'Model': f'Gradient Boosting (n={n_estimators}, lr={learning_rate})',
        'Test R2': gbr_r2,
        'Test RMSE': gbr_rmse
    }

    print(f"Gradient Boosting - Test R²: {gbr_r2:.4f}, Test RMSE: {gbr_rmse:,.2f}")
    
    return gbr_results, gbr_model

# ==========================================================================================================================================
# 5-Fold Cross-Validation, you are making sure that your $R^2$ scores weren't just a lucky break on one specific slice of data
# ==========================================================================================================================================
def run_cv_leaderboard(X, y, models_dict, cv_folds=5):
    """
    1. Runs cross-validation on a collection of models.
    2. Calculates the Mean R² and the Stability (Standard Deviation).
    3. Returns a clean DataFrame for comparison.
    """
    print(f"{cv_folds}-Fold Cross-Validation Results:")
    print("=" * 50)

    cv_summary = []

    for name, model in models_dict.items():
        # Perform CV
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2', n_jobs=-1)
        
        mean_score = scores.mean()
        std_dev = scores.std()
        
        cv_summary.append({
            'Model': name,
            'CV Mean R2': mean_score,
            'CV Std R2': std_dev,
            '95% Confidence': std_dev * 2  # The '+/-' range
        })
        
        print(f"{name:20}: R² = {mean_score:.4f} (+/- {std_dev*2:.4f})")

    # Convert to DataFrame and sort by the best performer
    cv_df = pd.DataFrame(cv_summary).sort_values(by='CV Mean R2', ascending=False)
    return cv_df

# ==================================================================================
# Feature Importance & Selection
# ==================================================================================
def plot_feature_importance(model, feature_names, model_name="Random Forest", top_n=20):
    """
    1. Extracts feature importance from a tree-based model.
    2. Creates a sorted DataFrame and a horizontal bar chart.
    3. Returns the importance DataFrame.
    """
    # 1. Extract Importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # 2. Slice for top_n to keep the chart readable
    plot_df = importance_df.head(top_n)

    # 3. Visualize
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=plot_df, 
        palette='viridis',
        hue='Feature',
        legend=False
    )
    
    plt.title(f'Top {top_n} Feature Importances ({model_name})')
    plt.xlabel('Importance Score (0.0 to 1.0)')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.show()

    return importance_df

# ==================================================================================
# Feature Importance & Selection
# ==================================================================================
def run_selected_feature_suite(X_train_scaled, X_test_scaled, y_train, y_test, selected_features):
    """
    1. Subsets the data to only include the most important features.
    2. Runs a battery of models to compare performance vs. the full dataset.
    3. Returns a comparison DataFrame.
    """
    # 1. Subset the data
    X_train_selected = X_train_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]
    
    print(f"Refining models using {len(selected_features)} selected features...")
    print("-" * 30)

    selected_results_list = []
    
    # 2. Define the "Finalist" models
    models_to_test = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge(alpha=1.0)),
        ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    ]

    # 3. Loop through and evaluate
    for name, model in models_to_test:
        # Using your existing evaluate_model function
        results, trained_model, _ = evaluate_model(
            model, 
            X_train_selected, 
            X_test_selected, 
            y_train, 
            y_test, 
            f"{name} (Selected)"
        )
        
        # Add a flag so we can identify these in the master list
        results['Feature Set'] = 'Selected'
        results['Model Name'] = name
        selected_results_list.append(results)
        
        print(f"{name:20}: Test R² = {results['Test R2']:.4f}")

    return pd.DataFrame(selected_results_list), X_train_selected.columns.tolist()

# ===========================================================================================================
# Evaluate the final model on the selected features and benchmark against the target's natural variance
# ===========================================================================================================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Standardizes the training, prediction, and metric calculation 
    to fit inside your Feature Selection loop.
    """
    # 1. Fit the model
    model.fit(X_train, y_train)
    
    # 2. Make predictions
    y_pred = model.predict(X_test)
    
    # 3. Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # 4. Format the results dictionary
    results = {
        'Model': model_name,
        'Test R2': r2,
        'Test RMSE': rmse,
        'Test MAE': mae
    }
    
    # Return the triplet your loop is looking for (results, model, predictions)
    return results, model, y_pred

# ==================================================================================
# Final Model Deployment
# ==================================================================================
def run_final_model_deployment(X_train_selected, X_test_selected, y_train, y_test, selected_features, target_range, target_std):
    """
    1. Trains the finalized Gradient Boosting model.
    2. Calculates comprehensive performance metrics.
    3. Benchmarks the error against the data's natural variance.
    """
    # 1. Initialize and Train
    final_model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3,
        random_state=42
    )
    final_model.fit(X_train_selected, y_train)
    
    # 2. Predict and Evaluate
    y_pred = final_model.predict(X_test_selected)
    
    final_r2 = r2_score(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_mae = mean_absolute_error(y_test, y_pred)
    
    # 3. Report
    print("=" * 50)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Algorithm:   Gradient Boosting Regressor")
    print(f"Features:    {len(selected_features)} Selected Columns")
    print(f"Test R²:     {final_r2:.4f}")
    print(f"Test RMSE:   {final_rmse:,.4f}")
    print(f"Test MAE:    {final_mae:,.4f}")
    print("-" * 50)
    
    # Benchmarking logic
    error_vs_range = (final_rmse / target_range) * 100
    error_vs_std = (final_rmse / target_std) * 100
    
    print(f"RMSE as % of target range: {error_vs_range:.1f}%")
    print(f"RMSE as % of target std:   {error_vs_std:.1f}%")
    
    return final_model, {
        'R2': final_r2, 
        'RMSE': final_rmse, 
        'MAE': final_mae,
        'Error_Range_Pct': error_vs_range
    }