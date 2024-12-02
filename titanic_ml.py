import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def feature_engineer(full_data):
    # Drop columns with excessive missing values
    threshold = 0.5 * len(full_data)  # Drop columns with >50% missing
    full_data = full_data.dropna(axis=1, thresh=threshold)

    # Drop duplicates and create combined age column
    full_data = full_data.drop_duplicates(subset='PassengerId')
    full_data['Age_combined'] = full_data['Age_wiki'].fillna(full_data['Age'])

    # Extract and categorize titles
    full_data['Title'] = full_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    def categorize_title(row):
        male_regular_titles = ['Mr', 'Master']
        male_rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir']
        female_regular_titles = ['Mrs', 'Miss']
        female_rare_titles = ['Lady', 'Dona', 'Countess', 'Mme', 'Mlle', 'Ms']

        if row['Sex'] == 'male':
            if row['Title'] in male_regular_titles:
                return 'Regular_Male'
            elif row['Title'] in male_rare_titles:
                return 'Rare_Male'
        elif row['Sex'] == 'female':
            if row['Title'] in female_regular_titles:
                return 'Regular_Female'
            elif row['Title'] in female_rare_titles:
                return 'Rare_Female'
        return 'Unknown'
    full_data['TitleCategory'] = full_data.apply(categorize_title, axis=1)

    # Add family features
    full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
    full_data['IsAlone'] = (full_data['FamilySize'] == 1).astype(int)

    # Fill missing values for Fare and categorical columns
    full_data['Fare'] = full_data['Fare'].fillna(full_data['Fare'].median())
    for col in ['Embarked', 'Hometown', 'Boarded', 'Destination', 'Class']:
        full_data[col] = full_data[col].fillna(full_data[col].mode()[0])


    # Define mapping for TitleCategory
    title_mapping = {
        'Regular_Male': 1,
        'Regular_Female': 2,
        'Rare_Male': 3,
        'Rare_Female': 4,
        'Unknown': 0  # Optionally handle unknown titles
    }
    full_data['TitleCategory'] = full_data['TitleCategory'].map(title_mapping)


    # Drop unnecessary columns
    columns_to_drop = ['Ticket', 'Name', 'Name_wiki', 'Age', 'Age_wiki', 'WikiId','Title']
    full_data = full_data.drop(columns=columns_to_drop)

    # Encode categorical features with Label Encoding
    label_encoder = LabelEncoder()
    for col in ['Destination', 'Hometown', 'Boarded', 'Sex', 'Embarked']:
        full_data[col] = label_encoder.fit_transform(full_data[col])

    # Handle missing ages
    if full_data['Age_combined'].isnull().sum() > 0:
        # Prepare data for training the regressor
        age_known = full_data[full_data['Age_combined'].notna()]
        age_missing = full_data[full_data['Age_combined'].isna()]

        # Include only numeric columns
        X = age_known.select_dtypes(include=["number"]).drop(columns=['Age_combined', 'PassengerId', 'Survived'], errors='ignore')
        y = age_known['Age_combined']
        X_missing = age_missing.select_dtypes(include=["number"]).drop(columns=['Age_combined', 'PassengerId', 'Survived'], errors='ignore')

        # Debug: Print column types
        print("X.dtypes before fitting RandomForest:")
        print(X.dtypes)

        # Train a Random Forest Regressor to predict missing ages
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Predict missing ages
        age_predictions = rf.predict(X_missing)
        full_data.loc[full_data['Age_combined'].isna(), 'Age_combined'] = age_predictions
    else:
        print("No missing ages to predict.")

    # Verify no remaining missing values
    if full_data.isnull().sum().sum() > 0:
        print("Warning: There are still missing values!")
    else:
        print("All missing values handled successfully.")
    # Discretize Age_combined
    age_bins = [-1, 12, 18, 35, 60, np.inf]  # Bins: Child, Teen, Adult, Middle-aged, Senior
    age_labels = ['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior']
    full_data['Age_bin'] = pd.cut(full_data['Age_combined'], bins=age_bins, labels=age_labels)

    # Discretize Fare
    fare_bins = [-1, 7.91, 14.45, 31.0, np.inf]  # Based on quantiles
    fare_labels = ['Low', 'Lower-Middle', 'Upper-Middle', 'High']
    full_data['Fare_bin'] = pd.cut(full_data['Fare'], bins=fare_bins, labels=fare_labels)



    # One-hot encode the binned columns if needed
    full_data = pd.get_dummies(full_data, columns=['Age_bin', 'Fare_bin'], drop_first=True)
    # Drop unnecessary columns
    columns_to_drop = ['Age_combined', 'Fare']
    full_data = full_data.drop(columns=columns_to_drop)

    return full_data

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def main():
    # Load datasets
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Feature engineering (assuming a feature_engineer function exists)
    train = feature_engineer(train)
    test = feature_engineer(test)

    # Split train data into features (X) and target (y)
    X = train.drop(columns=['PassengerId', 'Survived'])
    y = train['Survived']

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize XGBoost Classifier
    xgb_model = XGBClassifier(
        eval_metric='logloss', 
        random_state=42
    )

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, 10, 20]  # Adjust for imbalance
    }

    # GridSearchCV with Stratified K-Fold
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',  # Use ROC-AUC as the scoring metric
        cv=skf,
        n_jobs=-1,
        verbose=1
    )

    # Perform the grid search
    print("Performing grid search...")
    grid_search.fit(X, y)

    # Retrieve the best model and hyperparameters
    best_model = grid_search.best_estimator_
    print("\nBest Hyperparameters:\n", grid_search.best_params_)

    # Evaluate on validation data using Stratified K-Fold
    fold_accuracies = []
    fold_roc_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nEvaluating fold {fold + 1}...")

        # Split data into train and validation sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit the best model on this fold's training data
        best_model.fit(X_train, y_train)

        # Predict on the validation set
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]

        # Evaluate metrics
        fold_accuracy = accuracy_score(y_val, y_val_pred)
        fold_roc_auc = roc_auc_score(y_val, y_val_proba)

        fold_accuracies.append(fold_accuracy)
        fold_roc_aucs.append(fold_roc_auc)

        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
        print(f"Fold {fold + 1} ROC-AUC: {fold_roc_auc:.4f}")
        print(classification_report(y_val, y_val_pred))

    # Print average metrics across folds
    print(f"\nAverage Accuracy across folds: {np.mean(fold_accuracies):.4f}")
    print(f"Average ROC-AUC across folds: {np.mean(fold_roc_aucs):.4f}")

    # Fit the best model on the entire training set
    best_model.fit(X, y)

    # Predict survival for the test dataset
    X_test = test.drop(columns=['PassengerId'], errors='ignore')
    X_test = X_test.reindex(columns=X.columns, fill_value=0)  # Align columns with training data
    test['Survived'] = best_model.predict(X_test)

    # Save predictions concatenated with PassengerId to CSV
    submission = test[['PassengerId']].copy()
    submission['Survived'] = test['Survived'].astype(int)
    submission.to_csv("predicted_survival_xgboost_kfold.csv", index=False)

    print("\nPredictions saved to 'predicted_survival_xgboost_kfold.csv'.")

# Run the main function
if __name__ == "__main__":
    main()
