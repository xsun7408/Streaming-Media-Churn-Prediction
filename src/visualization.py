import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'clustered.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# 图1：Cluster vs Churn Rate
churn_rates = df.groupby('Cluster')['Churned'].mean()

plt.figure(figsize=(6, 4))
churn_rates.plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Cluster')
plt.tight_layout()

# 保存图片
plot_path = os.path.join(OUTPUT_DIR, 'churn_rate_by_cluster.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print("Plot saved to:", os.path.abspath(plot_path))


plt.show()