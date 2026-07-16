import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer

# ==============================================================================
# Configuration setup
# ==============================================================================
WORKSPACE_ROOT = Path(__file__).parent.resolve()
INPUT_FILE = WORKSPACE_ROOT / "output" / "merged" / "smart_mortality_merged.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "output" / "ml"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Initialize directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. Exploratory Data Analysis (EDA)
# ==============================================================================
def perform_eda(df):
    print("--- Starting Exploratory Data Analysis ---")
    print(f"Data dimensions: {df.shape}")
    
    # Target variable distribution (Death occurred vs No Death)
    # We will engineer a binary target: 1 if Deaths > 0 else 0
    df['Has_Death'] = (df['Deaths'] > 0).astype(int)
    
    print("\nTarget Variable Distribution (Has_Death):")
    print(df['Has_Death'].value_counts(normalize=True) * 100)
    
    # Plot 1: Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Has_Death')
    plt.title('Distribution of Households with Deaths')
    plt.xlabel('Has Death (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig(PLOTS_DIR / 'target_distribution.png')
    plt.close()
    
    # Plot 2: Deaths by Location (Top 15)
    location_deaths = df.groupby('location')['Has_Death'].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    location_deaths.plot(kind='bar', color='salmon')
    plt.title('Top 15 Locations by Proportion of Households with Deaths')
    plt.ylabel('Proportion of HHs with Death')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'deaths_by_location.png')
    plt.close()

    # Plot 3: Correlation Heatmap of numerical features
    numeric_cols = ['recall_period', 'Total', 'Births', 'Deaths', 'Joined', 'Left', 'Person_Time', 'Has_Death']
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'correlation_heatmap.png')
    plt.close()
    
    return df

# ==============================================================================
# 2. Data Preprocessing & Feature Engineering
# ==============================================================================
def preprocess_data(df):
    print("\n--- Data Preprocessing ---")
    
    # Create target
    y = df['Has_Death']
    
    # Select features
    # Drop columns that are perfectly correlated with target or not useful for prediction
    # E.g., 'Deaths', 'Deaths_U5', 'HH', 'Cluster', 'Team', 'Total_U5', 'Births_U5', 'Joined_U5', 'Left_U5', 'Person_Time_U5'
    features_to_drop = ['Deaths', 'Deaths_U5', 'HH', 'Cluster', 'Team', 'Has_Death', 
                        'Total_U5', 'Births_U5', 'Joined_U5', 'Left_U5', 'Person_Time_U5']
    X = df.drop(columns=features_to_drop)
    
    # Handle missing values in 'month' and 'year'
    imputer = SimpleImputer(strategy='median')
    X[['month', 'year']] = imputer.fit_transform(X[['month', 'year']])
    
    # Encode categorical features
    print("Encoding categorical variables: 'location', 'survey_type'")
    le_loc = LabelEncoder()
    le_type = LabelEncoder()
    
    X['location_encoded'] = le_loc.fit_transform(X['location'])
    X['survey_type_encoded'] = le_type.fit_transform(X['survey_type'])
    
    # Drop original categorical columns
    X = X.drop(columns=['location', 'survey_type'])
    
    return X, y, le_loc, le_type

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ==============================================================================
# 3. Model Training & Evaluation
# ==============================================================================
def train_and_evaluate(X, y):
    print("\n--- Model Training & Evaluation ---")
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', class_weight='balanced')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
    models.append(('NB', GaussianNB()))
    
    # Warning: SVM (SVC) is O(n^3) and will take a VERY long time on 34k rows.
    # We include it here as requested, but with a max_iter to prevent infinite hanging.
    models.append(('SVM', SVC(gamma='auto', max_iter=2000, class_weight='balanced')))
    models.append(('RF', RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)))
    
    # evaluate each model in turn
    results = []
    names = []
    print("\n--- Comparing Algorithms (5-Fold Cross Validation) ---")
    for name, model in models:
        try:
            kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1)
            results.append(cv_results)
            names.append(name)
            print('%s: ROC-AUC = %f (std: %f)' % (name, cv_results.mean(), cv_results.std()))
        except Exception as e:
            print(f"%s: Error - {e}" % name)

    # Plot Algorithm Comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot(results, tick_labels=names)
    plt.title('Algorithm Comparison (ROC-AUC Score)')
    plt.ylabel('ROC-AUC')
    plt.savefig(PLOTS_DIR / 'algorithm_comparison.png')
    plt.close()
    
    # Train the best overall model (Random Forest) on the full training set
    print("\nTraining final Random Forest Classifier...")
    best_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluation Metrics
    print("\nBest Model Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(PLOTS_DIR / 'confusion_matrix.png')
    plt.close()
    
    # Plot Feature Importance
    importances = best_model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_importance.png')
    plt.close()
    
    # Save test results
    test_results = pd.DataFrame({'True_Label': y_test, 'Predicted_Label': y_pred, 'Predicted_Prob': y_prob})
    test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    print(f"\nSaved plots to {PLOTS_DIR}")
    print(f"Saved test predictions to {OUTPUT_DIR}/test_predictions.csv")

def main():
    print("============================================================")
    print("Phase 4 & 5: Machine Learning and Model Assessment")
    print("============================================================")
    
    if not INPUT_FILE.exists():
        print(f"FATAL: The input dataset was not found at {INPUT_FILE}")
        return
        
    # Load data
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # EDA
    df = perform_eda(df)
    
    # Preprocessing
    X, y, le_loc, le_type = preprocess_data(df)
    
    # Train and Evaluate
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
