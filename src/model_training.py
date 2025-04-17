import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_config():
    """Load configuration from config.yaml"""
    with open('src/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_data(config):
    """Load the processed train and test data"""
    train_data = pd.read_csv(config['data']['train_data_path'])
    test_data = pd.read_csv(config['data']['test_data_path'])
    
    X_train = train_data.drop('price_category', axis=1)
    y_train = train_data['price_category']
    X_test = test_data.drop('price_category', axis=1)
    y_test = test_data['price_category']
    
    return X_train, X_test, y_train, y_test

def create_models(config):
    """Create instances of the classification models"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=config['model']['random_forest']['n_estimators'],
            max_depth=config['model']['random_forest']['max_depth'],
            random_state=config['model']['random_forest']['random_state']
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=config['model']['xgboost']['n_estimators'],
            max_depth=config['model']['xgboost']['max_depth'],
            learning_rate=config['model']['xgboost']['learning_rate'],
            random_state=config['model']['xgboost']['random_state']
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=config['model']['lightgbm']['n_estimators'],
            max_depth=config['model']['lightgbm']['max_depth'],
            learning_rate=config['model']['lightgbm']['learning_rate'],
            random_state=config['model']['lightgbm']['random_state']
        )
    }
    return models

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, config):
    """Train and evaluate all models"""
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=config['training']['cv_folds'],
            scoring=config['training']['scoring']
        )
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\nResults for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nClassification Report:")
        print(results[name]['classification_report'])
    
    return results

def plot_results(results):
    """Plot comparison of model performances"""
    # Prepare data for plotting
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    cv_means = [results[model]['cv_mean'] for model in models]
    cv_stds = [results[model]['cv_std'] for model in models]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
    plt.bar(x + width/2, cv_means, width, label='CV Mean Accuracy')
    
    # Add error bars for CV scores
    plt.errorbar(x + width/2, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    """Main function to run the model training pipeline"""
    # Load configuration
    config = load_config()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(config)
    
    # Create models
    models = create_models(config)
    
    # Train and evaluate models
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test, config)
    
    # Plot results
    plot_results(results)
    
    print("\nModel training and evaluation completed successfully!")
    print("Results have been saved to 'model_comparison.png'")

if __name__ == "__main__":
    main() 