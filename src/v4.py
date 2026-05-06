import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rules_path = os.path.join(
    BASE_DIR,
    'Data',
    'processed',
    'association_results',
    'cluster0_rules_to_churn.csv'
)

rules = pd.read_csv(rules_path)

plt.figure(figsize=(6, 5))

plt.scatter(
    rules['support'],
    rules['confidence'],
    s=rules['lift'] * 50,   # 点大小 = lift
    alpha=0.6
)

plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules Scatter Plot')

plt.tight_layout()

plot_path = os.path.join(BASE_DIR, 'results', 'association_scatter.png')
plt.savefig(plot_path, dpi=300)
plt.close()

print("Saved:", plot_path)