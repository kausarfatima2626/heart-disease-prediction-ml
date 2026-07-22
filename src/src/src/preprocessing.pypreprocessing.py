import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor():
    """
    Creates and returns a ColumnTransformer preprocessor pipeline.
    Fixes the label encoding bug on numerical features.
    """
    # 1. Define feature categories
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # 2. Define individual transformers
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 3. Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

if __name__ == "__main__":
    print("Preprocessing module initialized successfully!")
