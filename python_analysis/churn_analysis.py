import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Create folder
charts_path = "charts"
os.makedirs("outputs/reports", exist_ok=True)
# Function to save plots
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(charts_path, filename), dpi=300)
    plt.close()



#Here we load the dataset
try:
    df = pd.read_csv("data/indian_customer_churn_dataset_500.csv")
    print(" Dataset loaded successfully\n")
except FileNotFoundError:
    print(" Dataset not found! Check file path.")
    exit()

print(df.head())
print(df.tail())
#print(df.info())



#Data Cleaning
print("Data Cleaning in Process...\n")

#Converting date columns to datetime format
df['Signup_Date'] = pd.to_datetime(df['Signup_Date'])
df['Last_Active_Date'] = pd.to_datetime(df['Last_Active_Date'])
df['Churn_Date'] = pd.to_datetime(df['Churn_Date'], errors='coerce')
print(df.head())



#Handling Missing Values
df['Avg_Usage_Hours_per_Week'] = df['Avg_Usage_Hours_per_Week'].fillna(df['Avg_Usage_Hours_per_Week'].median())
print(df)


#Crating the Churn date 
df['Churn'] = df['Churn_Date'].notnull().astype(int)
print(df.head())

print("\nData Cleaning Completed Successfully\n")




#Feature Engineering
print("\n⚙️ Creating features...")

df['Customer_Lifetime'] = (df['Last_Active_Date'] - df['Signup_Date']).dt.days
df['Tenure_Months'] = df['Customer_Lifetime'] / 30
df['Signup_Month'] = df['Signup_Date'].dt.to_period('M').astype(str)

print("✅ Features created")



#save the cleaned dataset
# df.to_csv("data/cleaned_customer_churn_data.csv", index=False)

# print("✅ Cleaned dataset saved successfully!")





#Analysis and Visualization
print("\n Analysis and visualization in process...")

#churn Analysis
churn_rate = df['Churn'].mean()*100
print(churn_rate)


#Churn Analysis by plan
churn_by_plan = df.groupby('Plan_Type')['Churn'].mean()*100
print("Churn rate by plan: ", churn_by_plan)


#Churn Analysis by region
churn_by_region = df.groupby('Region')['Churn'].mean()*100
print("\n Churn rate by region; ", churn_by_region)



print(f"\n Overall Churn Rate: {churn_rate:.2f}%")
#print("\n Churn by Plan:\n", churn_by_plan)
#print("\n Churn by Region:\n", churn_by_region)

print("\n Analysis completed successfully")


#Visualization
print("\n visualization in process...")

# 1. Churn Distribution (Very Simple)
plt.figure()
df['Churn'].value_counts().plot(kind='bar')
plt.title("Churn Distribution", fontsize=14, fontweight='bold')
plt.xlabel("0 = Active, 1 = Churned")
plt.ylabel("Customers")
save_plot("churn_distribution.png")
plt.close()
print("Churn Distribution plot saved successfully")

# 2. Churn by Plan (Simple Average)
plan_churn = df.groupby('Plan_Type')['Churn'].mean()

plt.figure()
plan_churn.plot(kind='bar')
plt.title("Churn by Plan", fontsize=14, fontweight='bold')
plt.xlabel("Plan Type")
plt.ylabel("Churn Rate")
save_plot("churn_by_plan.png")
plt.close()
print("Churn by plan plot saved successfully")


# 3. Churn by Region (Simple View)
region_churn = df.groupby('Region')['Churn'].mean()

plt.figure()
region_churn.plot(kind='bar')
plt.title("Churn by Region", fontsize=14, fontweight='bold')
plt.xlabel("Region")
plt.ylabel("Churn Rate")
plt.xticks(rotation=45)
save_plot("churn_by_region.png")
plt.close()
print("Churn by Region plot saved successfully")

# 4. Customer Lifetime (Very Clean)
plt.figure()
plt.hist(df['Customer_Lifetime'], bins=20)
plt.title("Customer Lifetime")
plt.xlabel("Days")
plt.ylabel("Customers")
save_plot("customer_lifetime.png")
plt.close()
print("Customer Lifetime plot saved successfully")

# 5. Correlation Heatmap (Simplified)
plt.figure()
corr = df[['Age', 'Avg_Usage_Hours_per_Week', 'Customer_Lifetime', 'Churn']].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation")
save_plot("Correlation_heatmap.png")
plt.close()
print("Correlation Heatmap plot saved successfully")

print("\n All Visualizations completed and saved successfully")



#Cohort Analysis
print("\n📅 Performing cohort analysis...")

cohort = df.groupby(['Signup_Month', 'Churn']).size().unstack()

plt.figure()
cohort.plot(kind='bar', stacked=True)
plt.title("Cohort Analysis (Signup Month vs Churn)")
save_plot("cohort_analysis.png")
plt.close()

print("✅ Cohort analysis chart saved")




#save the cleaned dataset
df.to_csv("data/cleaned_customer_churn_data.csv", index=False)
print("\n Cleaned dataset saved in data/")



# FINAL MESSAGE

print("\n PROJECT COMPLETED SUCCESSFULLY!")
