import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'clustered.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# -------------------------
# 图1：Churn Rate by Cluster
# -------------------------
plt.figure(figsize=(6, 4))
churn_rates = df.groupby('Cluster')['Churned'].mean()
churn_rates.plot(kind='bar')

plt.xlabel('Cluster')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Cluster')
plt.tight_layout()

plot_path1 = os.path.join(OUTPUT_DIR, 'churn_rate_by_cluster.png')
plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
plt.close()

print("Saved:", os.path.abspath(plot_path1))

# -------------------------
# 图2：Feature Comparison by Cluster
# -------------------------
plt.figure(figsize=(7, 4))
feature_means = df.groupby('Cluster')[['Satisfaction_Score', 'Monthly_Spend']].mean()
feature_means.plot(kind='bar')

plt.xlabel('Cluster')
plt.ylabel('Average Value')
plt.title('Feature Comparison by Cluster')
plt.tight_layout()

plot_path2 = os.path.join(OUTPUT_DIR, 'feature_comparison.png')
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
plt.close()

print("Saved:", os.path.abspath(plot_path2))