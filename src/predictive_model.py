"""
Predictive Modeling Module.
Trains a Random Forest Regressor framework to impute missing mechanical properties 
using advanced multivariate pattern recognition.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class MaterialPropertyPredictor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = RandomForestRegressor(n_estimators=150, random_state=42)
        
    def predict_missing_fatigue_strength(self) -> pd.DataFrame:
        """
        Derives missing fatigue strength values based on the inherent correlation 
        with tensile strength, yield strength, and superficial hardness.
        """
        df = self.data.copy()
        target_col = 'fatigue_strength_mpa'
        feature_cols = ['yield_strength_mpa', 'tensile_strength_mpa', 'hardness_hv']
        
        if not all(col in df.columns for col in feature_cols + [target_col]):
            return df
            
        # Segment the dataset into a complete training set and an inference set
        known_mask = df[target_col].notna() & df[feature_cols].notna().all(axis=1)
        unknown_mask = df[target_col].isna() & df[feature_cols].notna().all(axis=1)
        
        if known_mask.sum() < 15:
            # Insufficient samples to train a statistically reliable regression manifold
            return df
            
        X_train = df.loc[known_mask, feature_cols]
        y_train = df.loc[known_mask, target_col]
        X_infer = df.loc[unknown_mask, feature_cols]
        
        self.model.fit(X_train, y_train)
        
        if len(X_infer) > 0:
            predictions = self.model.predict(X_infer)
            df.loc[unknown_mask, target_col] = predictions
            
        return df