import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 路径设置（必须有）
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 读取 association 结果
# =========================
rules_path = os.path.join(
    BASE_DIR,
    'Data',
    'processed',
    'association_results',
    'cluster0_rules_to_churn.csv'
)

rules = pd.read_csv(rules_path)

# =========================
# 选 Top 10
# =========================
top_rules = rules.sort_values(by='lift', ascending=False).head(10)

# =========================
# 把 frozenset 转成人话
# =========================
def format_rule(row):
    antecedent = str(row['antecedents']) \
        .replace("frozenset({", "") \
        .replace("})", "")
    return antecedent

labels = top_rules.apply(format_rule, axis=1)

# =========================
# 画图
# =========================
plt.figure(figsize=(8, 5))
plt.barh(range(len(top_rules)), top_rules['lift'])

plt.yticks(range(len(top_rules)), labels)
plt.xlabel('Lift')
plt.title('Top Churn-Related Feature Combinations')
plt.gca().invert_yaxis()
plt.tight_layout()

# =========================
# 保存
# =========================
plot_path = os.path.join(OUTPUT_DIR, 'top_churn_rules.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print("Saved:", os.path.abspath(plot_path))