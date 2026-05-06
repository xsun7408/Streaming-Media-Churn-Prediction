import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 路径设置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 读取 无Satisfaction 的规则
# =========================
rules_path = os.path.join(
    BASE_DIR,
    'Data',
    'processed',
    'association_results_no_satisfaction',
    'cluster0_no_satisfaction_rules_to_churn.csv'
)

rules = pd.read_csv(rules_path)

# =========================
# 选 Top 8（更清晰）
# =========================
top_rules = rules.sort_values(by='lift', ascending=False).head(8)

# =========================
# 清理 frozenset（变成人话）
# =========================
def clean_text(x):
    return str(x).replace("frozenset({", "").replace("})", "")

labels = top_rules['antecedents'].apply(clean_text)

# =========================
# 画图
# =========================
plt.figure(figsize=(8, 5))

plt.barh(range(len(top_rules)), top_rules['lift'])

plt.yticks(range(len(top_rules)), labels)
plt.xlabel('Lift')
plt.title('Churn Drivers Without Satisfaction')

plt.gca().invert_yaxis()
plt.tight_layout()

# =========================
# 保存
# =========================
plot_path = os.path.join(OUTPUT_DIR, 'churn_without_satisfaction.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print("Saved:", os.path.abspath(plot_path))