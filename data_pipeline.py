import pandas as pd
import numpy as np
import os

def run_data_pipeline(file_path):
    print("=" * 70)
    print("PALO ALTO NETWORKS - HR ANALYTICS PIPELINE")
    print("=" * 70)
    
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        print(f"❌ Error: '{file_path}' is missing or empty.")
        return None

    df = pd.read_csv(file_path)
    
    # 1. Validation & Normalization
    satisfaction_columns = [
        'EnvironmentSatisfaction', 'JobSatisfaction', 
        'RelationshipSatisfaction', 'WorkLifeBalance', 'JobInvolvement'
    ]
    for col in satisfaction_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].clip(1, 4)
        df[f"{col}_Normalized"] = (df[col] - 1) / (4 - 1)

    # 2. Engagement Index & Stability Score
    engagement_cols = [
        'JobInvolvement_Normalized', 'JobSatisfaction_Normalized',
        'EnvironmentSatisfaction_Normalized', 'RelationshipSatisfaction_Normalized'
    ]
    df['Engagement_Index'] = df[engagement_cols].mean(axis=1) * 100
    
    # Technical Document KPI: Satisfaction Stability Score
    df['Satisfaction_Stability_Score'] = (1 - df[engagement_cols].std(axis=1)) * 100

    # 3. Burnout Risk & Workload
    def assign_burnout_risk(row):
        overtime = row['OverTime'] == 'Yes'
        low_wlb = row['WorkLifeBalance'] <= 2
        if overtime and low_wlb:
            return 'High'
        elif overtime or low_wlb:
            return 'Medium'
        else:
            return 'Low'

    df['Burnout_Risk_Level'] = df.apply(assign_burnout_risk, axis=1)
    
    # FIX: Indentation corrected for np.where
    df['Burnout_Risk_Score'] = np.where(
        df['Burnout_Risk_Level'] == 'High', 100,
        np.where(df['Burnout_Risk_Level'] == 'Medium', 50, 0)
    )

    # Technical Document KPI: Workload Stress Indicator
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    df['Travel_Intensity'] = df['BusinessTravel'].map(travel_map)
    df['Overtime_Intensity'] = np.where(df['OverTime'] == 'Yes', 2, 0)
    df['Workload_Stress_Indicator'] = df['Travel_Intensity'] + df['Overtime_Intensity']

    # SAVE FRESH DATA
    processed_path = "Palo_Alto_Processed.csv"
    df.to_csv(processed_path, index=False)
    print("✔ Pipeline executed successfully! All 5 KPIs generated and saved securely.")
    return df

if __name__ == "__main__":
    run_data_pipeline("Palo Alto Networks.csv")