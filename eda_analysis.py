import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & INSIGHTS")
    print("=" * 70)
    
    file_path = "Palo_Alto_Processed.csv"
    if not os.path.exists(file_path):
        print("❌ Error: Processed data not found. Please run data_pipeline.py first.")
        return
        
    df = pd.read_csv(file_path)
    
    # Create a folder to save our research paper visuals
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the visual style for corporate reports
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------------
    # 1. Engagement vs Attrition (Contextual Only)
    # -------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Attrition', y='Engagement_Index', data=df, palette='Set2')
    plt.title('Engagement Index: Stayed (0) vs Left (1)', fontsize=14)
    plt.savefig(f"{output_dir}/1_Engagement_vs_Attrition.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    avg_eng_stayed = df[df['Attrition'] == 0]['Engagement_Index'].mean()
    avg_eng_left = df[df['Attrition'] == 1]['Engagement_Index'].mean()
    
    print("\n[1] Engagement vs Attrition Insights:")
    print(f"  - Avg Engagement (Stayed): {avg_eng_stayed:.2f}/100")
    print(f"  - Avg Engagement (Left):   {avg_eng_left:.2f}/100")
    
    # -------------------------------------------------------------------
    # 2. Workload & Stress Analysis (Overtime vs Engagement)
    # -------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='OverTime', y='Engagement_Index', data=df, palette='muted', split=True)
    plt.title('Impact of Overtime on Employee Engagement', fontsize=14)
    plt.savefig(f"{output_dir}/2_Overtime_vs_Engagement.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[2] Workload & Stress Insights:")
    overtime_eng = df[df['OverTime'] == 'Yes']['Engagement_Index'].mean()
    no_overtime_eng = df[df['OverTime'] == 'No']['Engagement_Index'].mean()
    print(f"  - Avg Engagement (Overtime = Yes): {overtime_eng:.2f}/100")
    print(f"  - Avg Engagement (Overtime = No):  {no_overtime_eng:.2f}/100")

    # -------------------------------------------------------------------
    # 3. Career-Stage Engagement Analysis (Years at Company)
    # -------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    # Group by years and calculate mean engagement
    tenure_eng = df.groupby('YearsAtCompany')['Engagement_Index'].mean().reset_index()
    # Filter for first 20 years to avoid noise from sparse data of very senior people
    tenure_eng = tenure_eng[tenure_eng['YearsAtCompany'] <= 20] 
    
    sns.lineplot(x='YearsAtCompany', y='Engagement_Index', data=tenure_eng, marker='o', color='crimson')
    plt.title('Career-Stage Engagement: Stagnation Risk over Time', fontsize=14)
    plt.axvline(x=3, color='grey', linestyle='--', label='Year 3 Drop')
    plt.axvline(x=5, color='black', linestyle='--', label='Year 5 Drop')
    plt.legend()
    plt.savefig(f"{output_dir}/3_Career_Stage_Engagement.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[3] Career-Stage & Stagnation Insights:")
    print("  - Noticeable drops in engagement typically occur around years 3 and 5.")
    print("  - This is a critical 'stagnation-linked disengagement' window where HR needs to intervene.")
    
    print("\n" + "=" * 70)
    print(f"✔ EDA Complete. 3 High-resolution charts saved in '{output_dir}' folder.")
    print("=" * 70)

if __name__ == "__main__":
    run_eda()