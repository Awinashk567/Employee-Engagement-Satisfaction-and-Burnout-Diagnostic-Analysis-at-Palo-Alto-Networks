import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set standard premium page config
st.set_page_config(page_title="Palo Alto HR Diagnostics", layout="wide", page_icon="🛡️")

# Custom CSS to make it look professional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1e3a8a; }
    h3 { color: #1e40af; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Palo_Alto_Processed.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Processed data file not found. Please run data_pipeline.py first.")
    st.stop()

# Main Title & Background Context
st.title("🛡️ Palo Alto Networks: HR Diagnostic Dashboard")
st.markdown("### *Preventive Employee Experience & Early-Warning System*")

st.divider()

# --- SIDEBAR: CONTROL PANEL (USER CAPABILITIES) ---
with st.sidebar:
    st.header("🎯 Control Panel & Filters")

    # Defensive string casting to prevent sorting errors
    all_departments = sorted(df['Department'].dropna().astype(str).unique())
    selected_depts = st.multiselect("Select Departments", all_departments, default=all_departments)

    all_roles = sorted(df['JobRole'].dropna().astype(str).unique())
    selected_roles = st.multiselect("Select Job Roles", all_roles, default=all_roles)

    overtime_filter = st.radio("Overtime Status Toggle", ["All Employees", "Yes Only", "No Only"])

    eng_threshold = st.slider("Low Engagement Alert Threshold", 0, 100, 55)

    max_tenure = int(pd.to_numeric(df['YearsAtCompany'], errors='coerce').max())
    tenure_range = st.slider("Employee Tenure Range (Years at Company)", 0, max_tenure, (0, max_tenure))

# Apply Filter Logic defensively
filtered_df = df[
    (df['Department'].isin(selected_depts)) &
    (df['JobRole'].isin(selected_roles)) &
    (pd.to_numeric(df['YearsAtCompany'], errors='coerce') >= tenure_range[0]) &
    (pd.to_numeric(df['YearsAtCompany'], errors='coerce') <= tenure_range[1])
]

if overtime_filter == "Yes Only":
    filtered_df = filtered_df[filtered_df['OverTime'] == 'Yes']
elif overtime_filter == "No Only":
    filtered_df = filtered_df[filtered_df['OverTime'] == 'No']

if filtered_df.empty:
    st.warning("⚠️ No employee records match the selected filter criteria. Please broaden your selection.")
    st.stop()


# --- MODULE 1: ENGAGEMENT HEALTH OVERVIEW (ALL 5 KPIs) ---
st.markdown("### 📊 Organization-Wide Health KPIs")

# 5 Columns generated guaranteed
m1, m2, m3, m4, m5 = st.columns(5)

avg_eng = pd.to_numeric(filtered_df['Engagement_Index'], errors='coerce').mean()
m1.metric("Org Engagement Index", f"{avg_eng:.1f} / 100")

high_burnout_count = len(filtered_df[filtered_df['Burnout_Risk_Level'] == 'High'])
m2.metric("Burnout Risk Score (High)", f"{high_burnout_count} Personnel", delta_color="inverse")

# FIX: Super Defensive logic to completely avoid string/int Division TypeErrors
wlb_series = filtered_df['WorkLifeBalance']
if wlb_series.dtype == object:
    reverse_wlb_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    wlb_numeric = wlb_series.map(reverse_wlb_map).fillna(pd.to_numeric(wlb_series, errors='coerce'))
else:
    wlb_numeric = pd.to_numeric(wlb_series, errors='coerce')
    
wlb_index = (wlb_numeric.mean() / 4) * 100
m3.metric("Work-Life Balance Index", f"{wlb_index:.1f}%")

avg_stress = pd.to_numeric(filtered_df['Workload_Stress_Indicator'], errors='coerce').mean()
m4.metric("Workload Stress", f"{avg_stress:.2f} / 4.0")

if 'Satisfaction_Stability_Score' in filtered_df.columns:
    avg_stability = pd.to_numeric(filtered_df['Satisfaction_Stability_Score'], errors='coerce').mean()
    m5.metric("Satisfaction Stability", f"{avg_stability:.1f}%")
else:
    m5.metric("Satisfaction Stability", "N/A")

st.markdown("<br>", unsafe_allow_html=True)

# --- MODULE 2 & 3: ADVANCED VISUALIZATIONS ---
tab1, tab2, tab3 = st.tabs(["🔥 Burnout & Stress", "📈 Engagement & Stagnation", "💼 Role & Level Analysis"])

with tab1:
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        st.markdown("#### Burnout Risk Segments")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        risk_data = filtered_df['Burnout_Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        sns.barplot(x=risk_data.index, y=risk_data.values, hue=risk_data.index, palette={'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}, ax=ax1, legend=False)
        ax1.set_ylabel("Employee Count")
        st.pyplot(fig1)
        
    with col_sub2:
        st.markdown("#### Overtime vs Work-Life Imbalance")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='WorkLifeBalance', hue='OverTime', data=filtered_df, palette='Set2', ax=ax2)
        ax2.set_title("Work-Life Balance Ratings by Overtime Status")
        st.pyplot(fig2)

with tab2:
    col_sub3, col_sub4 = st.columns(2)
    with col_sub3:
        st.markdown("#### Tenure vs Engagement Trends")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        tenure_trends = filtered_df.groupby('YearsAtCompany')['Engagement_Index'].mean().reset_index()
        sns.lineplot(x='YearsAtCompany', y='Engagement_Index', data=tenure_trends, marker='o', color='#1e3a8a', ax=ax3)
        st.pyplot(fig3)
        
    with col_sub4:
        st.markdown("#### Job vs Environment Satisfaction")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        
        # Defensive numeric casting for KDE Plot
        job_sat = pd.to_numeric(filtered_df['JobSatisfaction'], errors='coerce').dropna()
        env_sat = pd.to_numeric(filtered_df['EnvironmentSatisfaction'], errors='coerce').dropna()
        
        sns.kdeplot(job_sat, label='Job Satisfaction', fill=True, color='purple', ax=ax4)
        sns.kdeplot(env_sat, label='Environment Satisfaction', fill=True, color='blue', ax=ax4)
        ax4.legend()
        st.pyplot(fig4)

with tab3:
    st.markdown("#### Engagement Heatmap: Job Role vs Job Level")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    
    heatmap_df = filtered_df.copy()
    heatmap_df['Engagement_Index'] = pd.to_numeric(heatmap_df['Engagement_Index'], errors='coerce')
    
    role_level_pivot = heatmap_df.pivot_table(index='JobRole', columns='JobLevel', values='Engagement_Index', aggfunc='mean')
    sns.heatmap(role_level_pivot, annot=True, cmap='YlGnBu', fmt=".1f", linewidths=.5, ax=ax5)
    st.pyplot(fig5)

st.divider()

# --- MODULE 4: MANAGER ACTION PANEL ---
st.markdown("### 🚨 Manager Action Panel")
st.markdown("Employees matching low-engagement thresholds or high burnout parameters.")

action_needed_df = filtered_df[
    (pd.to_numeric(filtered_df['Engagement_Index'], errors='coerce') < eng_threshold) | 
    (filtered_df['Burnout_Risk_Level'] == 'High')
]

presentation_cols = ['Department', 'JobRole', 'JobLevel', 'YearsAtCompany', 'Engagement_Index', 'Burnout_Risk_Level', 'OverTime', 'WorkLifeBalance']
display_action_df = action_needed_df[presentation_cols].copy()
display_action_df['Engagement_Index'] = pd.to_numeric(display_action_df['Engagement_Index'], errors='coerce').round(1)

# UX Improvement: Map numerical values to human-readable labels securely
job_level_map = {1: 'Entry Level', 2: 'Junior', 3: 'Mid-Level', 4: 'Senior', 5: 'Executive'}
wlb_map = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}

display_action_df['JobLevel'] = pd.to_numeric(display_action_df['JobLevel'], errors='coerce').map(job_level_map).fillna(display_action_df['JobLevel'])
display_action_df['WorkLifeBalance'] = pd.to_numeric(display_action_df['WorkLifeBalance'], errors='coerce').map(wlb_map).fillna(display_action_df['WorkLifeBalance'])

# Row-wise highlighting logic
def highlight_entire_row(row):
    if row['Burnout_Risk_Level'] == 'High':
        return ['background-color: #ffcccc; color: #900000; font-weight: bold;'] * len(row)
    elif row['Burnout_Risk_Level'] == 'Medium':
        return ['background-color: #ffffcc; color: #806000;'] * len(row)
    return [''] * len(row)

# Apply row-wise styling
styled_df = display_action_df.style.apply(highlight_entire_row, axis=1)

# Render the dataframe
st.dataframe(styled_df, use_container_width=True, hide_index=True) # type: ignore

st.markdown("<br>", unsafe_allow_html=True)

# Export Button & Email Template Layout
col_btn, col_mail = st.columns([1, 1.5])

with col_btn:
    st.markdown("#### 📥 Export Data")
    st.write("Download the filtered list to distribute to HR business partners.")
    csv = display_action_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Priority Intervention List (CSV)",
        data=csv,
        file_name='HR_Intervention_List.csv',
        mime='text/csv',
    )

with col_mail:
    st.markdown("#### ✉️ Automated 1-on-1 Email Template")
    st.write("Use the copy button in the top right corner of the box below to quickly draft an email to at-risk employees.")
    
    email_template = """Subject: Checking in - Let's grab a coffee ☕\n\nHi [Employee Name],\n\nI was looking at our recent team workload trends and wanted to check in. It looks like you've been putting in a lot of effort lately, and I want to make sure we are actively supporting your work-life balance. \n\nYour well-being is a priority for us at Palo Alto Networks. Let's schedule a brief 1-on-1 this week to discuss your current workload and see how we can better support you.\n\nBest regards,\n[Your Name / HR Partner]"""
    
    st.code(email_template, language='text')