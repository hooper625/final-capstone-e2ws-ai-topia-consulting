"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, preprocess, split_data
"""
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pipelines.data_pipeline import create_temporal_features

from collections import Counter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ===================================================================================================================================
# Processing & Feature Engineering for 311 Service Request Data
# ===================================================================================================================================
def complaints_engineer_features(df, text_col='complaint_type', date_col='created_date', desc_col='Description', drop_cols=None):
    """
    Create new features from existing columns.
    
    Args:
        df (pd.DataFrame): The input data.
        text_col (str): Column for category grouping (default: 'complaint_type').
        date_col (str): Column for time features (default: 'created_date').
        desc_col (str): Column for word count features (default: 'Description').
        drop_cols (list): The list of columns you want to remove.
    """
    
    # Basic Cleaning & Dynamic Dropping
    # If the other component sends a list, we use it. Otherwise, we do nothing.
    if drop_cols:
        print(f"Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols, errors='ignore').copy()

    # Group low frequency complaint types into 'Other'
    if text_col in df.columns:
        df = group_low_frequency_categories(df, text_col, top_n=24, other_label="Other")

    # Create temporal features (hour, day, etc.)
    if date_col in df.columns:
        df = create_temporal_features(df) 
        
    # Extract Top 20 Word Features from Description
    if desc_col in df.columns:
        df = description_word_count(df)
        
    return df

# =============================================================================
# Identify top 20 words in the Description column and features for each
# =============================================================================
def description_word_count(df):
    """Add features for the top 20 words in the Description column."""
    if 'Description' in df.columns:
        # Combine all descriptions, lowercase them, and find all words
        all_text = ' '.join(df['Description'].dropna()).lower()
        words = re.findall(r'\w+', all_text)
        
        # Filter out stop words
        stop_words = {'on', 'at', 'the', 'and', 'of', 'to', 'in', 'from', 'near', 'i', 'rd', 'st', 'ave', 'blvd', 'hwy', 'highway', 'street', 'road', 'due', 'us', 'ca', 'la', 'ny', 'tx', 'fl', 'il', 'wa', 'pa', 'oh', 'mi', 'ga', 'nc', 'nj', 'va', 'ma', 'az', 'co', 'nv', 'with', 'dr', 's', 'n', 'e', 'w', '95'}
        keywords = Counter([w for w in words if w not in stop_words])
        
        # Get top 20 words
        top_20_words = [word for word, count in keywords.most_common(20)]
        
        # Create columns for each top word - count occurrences in each description
        for word in top_20_words:
            df[f'word_{word}'] = df['Description'].fillna('').apply(lambda x: x.lower().count(word))
        
        # Drop the Description column
        df = df.drop(columns=['Description'], errors='ignore')
    
    return df

# =============================================================================
# Applies Clean text logic to selected columns
# =============================================================================
def clean_selected_columns(df, columns_to_clean):
    """
    Takes a DataFrame and a list of column names, 
    applying the clean_text logic only to those columns.
    """
    df_cleaned = df.copy()
    
    for col in columns_to_clean:
        if col in df_cleaned.columns:
            print(f"Cleaning text in column: {col}...")
            # We cast to string to handle any 'NaN' or numeric values safely
            df_cleaned[col] = df_cleaned[col].astype(str).apply(clean_text)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            
    return df_cleaned

# =============================================================================
# Applies Clean text logic to selected columns
# =============================================================================
def clean_text(text):
    """Clean a single string."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Handle empty values or pandas NaNs
    if pd.isna(text) or text.lower() == 'nan' or text.strip() == '':
        return ""
    
    # Lowercase & ASCII only (removes emojis)
    text = text.lower().encode('ascii', 'ignore').decode('ascii')
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize & Lemmatize
    words = text.split()

    # Ensure lemmatizer and stop_words are defined in your environment
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words and len(word) > 1]

    return ' '.join(words)

# =============================================================================
# Group into the smaller categories
# =============================================================================
def group_low_frequency_categories(df, column, top_n=24, other_label="Other"):
    """
    Groups all values outside the top_n into a single 'Other' category.
    Works dynamically for any categorical column.
    """
    # Identify the top N categories
    top_categories = df[column].value_counts().nlargest(top_n).index
    
    # Map values: if it's in top_categories, keep it; otherwise, use 'Other'
    df[column] = df[column].where(df[column].isin(top_categories), other_label)
    
    print(f"Column '{column}' grouped: {top_n} categories kept + '{other_label}'")
    return df

# =============================================================================
# Get top words by category
# =============================================================================
def get_top_words_by_category(df, label_col, text_col, top_n=20):
    """
    Dynamically loops through unique labels in label_col and 
    finds the most common words in text_col for each.
    """
    TEXT_COLS= [text_col]  # List of columns to clean
    df = clean_selected_columns(df, TEXT_COLS)

    # Get the unique categories (e.g., ['positive', 'negative', 'neutral'])
    categories = df[label_col].unique()
    
    results = {}

    for category in categories:
        # Filter for this specific category
        # We ensure we only take non-null strings
        mask = (df[label_col] == category) & (df[text_col].notna())
        category_data = df.loc[mask, text_col]
        
        # Combine all texts and split into words
        # .str.cat joins the whole series efficiently
        all_text = category_data.str.cat(sep=' ')
        all_words = all_text.split()
        
        # Count frequencies
        top_words = Counter(all_words).most_common(top_n)
        results[category] = top_words
        
        # Print results in a clean format
        print(f'\n=== {str(category).upper()} (top {top_n} words) ===')
        if not top_words:
            print("  No words found for this category.")
        for word, count in top_words:
            print(f'  {word:20s} {count}')
            
    return results

# =============================================================================
# Length Normalization
# =============================================================================
def get_sequence_stats(df, text_col):
    """
    Calculates length stats to help set the 'max_length' for the NN.
    This helps Neural networks usually require input sequences of the same length
    it's helpful to know the distribution of your text lengths 
    so you can decide where to "cut" or "pad.
    """
    lengths = df[text_col].str.split().str.len()
    print(f"Average Length: {lengths.mean():.2f}")
    print(f"95th Percentile: {lengths.quantile(0.95):.2f}") # Use this for Padding
    return lengths

# =============================================================================
#Add an "Unknown" Tokenizer (OOV Handling)
# =============================================================================
def handle_rare_words(df, text_col, min_freq=2):
    """Replaces words that appear very rarely with an 'UNK' token."""
    all_words = ' '.join(df[text_col]).split()
    counts = Counter(all_words)
    
    # Identify rare words
    rare_words = {word for word, count in counts.items() if count < min_freq}
    
    # Replace them
    df[text_col] = df[text_col].apply(lambda x: ' '.join(
        [word if word not in rare_words else 'UNK' for word in x.split()]
    ))
    return df
# =============================================================================
# Core Embedding Function
# =============================================================================
def get_document_embedding(text, model, vector_size=100):
    """
    Finds the average vector for a piece of text.
    Works whether you pass a full Gensim model or just the '.wv' part.
    """
    # Convert to string (to avoid errors with NaNs) and split
    words = str(text).split()
    
    # Dynamic check: Get the vectors regardless of how the model was loaded
    # This is the line that uses 'wv' as a string to look for the attribute
    wv = model.wv if hasattr(model, 'wv') else model

    # Only grab vectors for words that actually exist in the model
    valid_vectors = [wv[w] for w in words if w in wv]

    # Return zeros if no words found, otherwise return the average
    if not valid_vectors:
        return np.zeros(vector_size)
    
    return np.mean(valid_vectors, axis=0)


