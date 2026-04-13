"""
Recommendation Engine Module for the Smart Alloy Selector.
Applies discrete constraint filtering and the TOPSIS algorithm for continuous
multi-dimensional property optimization.
"""

import pandas as pd
import numpy as np


class MaterialRecommender:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def filter_hard_constraints(self, constraints_dict: dict) -> pd.DataFrame:
        """
        Excludes candidate materials that absolutely violate the user's defined limits.
        """
        df = self.data.copy()

        if constraints_dict.get('is_3d_printable') is not None:
            req_printable = 'Yes' if constraints_dict['is_3d_printable'] else 'No'
            df = df[df['printable_3d'] == req_printable]

        if constraints_dict.get('max_density_g_cm3') is not None:
            df = df[df['density_g_cm3'] <= constraints_dict['max_density_g_cm3']]

        if constraints_dict.get('min_yield_strength_mpa') is not None:
            df = df[df['yield_strength_mpa'] >= constraints_dict['min_yield_strength_mpa']]

        if constraints_dict.get('min_service_temp_c') is not None:
            df = df[df['max_service_temp_c'] >= constraints_dict['min_service_temp_c']]

        if constraints_dict.get('max_cost_usd_kg') is not None:
            df = df[df['cost_usd_kg'] <= constraints_dict['max_cost_usd_kg']]

        if constraints_dict.get('must_be_corrosion_resistant') is True:
            df = df[df['corrosion_resistance'] >= 3.0]

        if constraints_dict.get('preferred_category') is not None:
            category = constraints_dict['preferred_category'].lower()
            df = df[df['category'].str.lower().str.contains(category, na=False)]

        return df

    def apply_topsis(
        self,
        df: pd.DataFrame,
        criteria_cols: list,
        weights: list,
        impacts: list
    ) -> pd.DataFrame:
        """
        Implements the Technique for Order of Preference by Similarity to Ideal Solution.
        """
        if len(df) <= 1:
            if len(df) == 1:
                df = df.copy()
                df['topsis_score'] = 1.0
            return df

        # Step 1: Extract decision matrix
        X = df[criteria_cols].values.astype(float)

        # Step 2: Normalize matrix
        norm_X = X / np.sqrt((X ** 2).sum(axis=0) + 1e-12)

        # Step 3: Weighted normalized matrix
        weighted_X = norm_X * weights

        # Step 4: Compute ideal best / worst
        ideal_best = np.zeros(len(criteria_cols))
        ideal_worst = np.zeros(len(criteria_cols))

        for i, impact in enumerate(impacts):
            if impact == '+':
                ideal_best[i] = np.max(weighted_X[:, i])
                ideal_worst[i] = np.min(weighted_X[:, i])
            else:
                ideal_best[i] = np.min(weighted_X[:, i])
                ideal_worst[i] = np.max(weighted_X[:, i])

        # Step 5: Distance to ideal solutions
        dist_best = np.sqrt(((weighted_X - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_X - ideal_worst) ** 2).sum(axis=1))

        # Step 6: Compute TOPSIS score
        scores = dist_worst / (dist_best + dist_worst + 1e-12)

        df_scored = df.copy()
        df_scored['topsis_score'] = scores

        # Engineering realism penalty layer
        penalty_map = {
            'Amorphous Metal': 0.80,
            'Carbon': 0.65,
            'Ceramic': 0.70,
            'Semiconductor': 0.60
        }

        for category, penalty in penalty_map.items():
            df_scored.loc[
                df_scored['category'] == category,
                'topsis_score'
            ] *= penalty

        # Final ranking
        df_scored = df_scored.sort_values(
            by='topsis_score',
            ascending=False
        )

        return df_scored

    def get_recommendations(self, parsed_constraints: dict) -> pd.DataFrame:
        """
        Combines deterministic filtering with multi-objective geometric ranking.
        """
        filtered_df = self.filter_hard_constraints(parsed_constraints)

        if len(filtered_df) == 0:
            return filtered_df

        # Expanded engineering criteria set
        criteria_cols = [
            'yield_strength_mpa',
            'density_g_cm3',
            'max_service_temp_c',
            'cost_usd_kg',
            'fatigue_strength_mpa',
            'elastic_modulus_gpa',
            'corrosion_resistance',
            'machinability',
            'weldability'
        ]

        impacts = [
            '+',   # yield strength
            '-',   # density
            '+',   # service temp
            '-',   # cost
            '+',   # fatigue strength
            '+',   # stiffness
            '+',   # corrosion resistance
            '+',   # machinability
            '+'    # weldability
        ]

        # Balanced engineering weights
        weights = [
            0.20,
            0.15,
            0.10,
            0.10,
            0.15,
            0.10,
            0.08,
            0.06,
            0.06
        ]

        ranked_df = self.apply_topsis(
            filtered_df,
            criteria_cols,
            weights,
            impacts
        )

        return ranked_df