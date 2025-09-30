### Plot the decoding accuracy across time bins with a boxplot and mean line
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'


# Load the CSV file
csv_path = r"S:\GVuilleumier\GVuilleumier\groups\jies\Spatial_Navigation\Final_data\GLM\MVPA_ACC\group_accuracy_SPC.csv"
df = pd.read_csv(csv_path)

# Reshape for plotting
df_melt = df.melt(id_vars="subject", var_name="Bin", value_name="Accuracy")

# Plot
plt.figure(figsize=(12, 4))
sns.boxplot(data=df_melt, x="Bin", y="Accuracy", color='white', showfliers=False)
sns.stripplot(data=df_melt, x="Bin", y="Accuracy", color='black',
                jitter=True, size=6, alpha=0.4, edgecolor='gray', linewidth=0.5)

mean_acc = df_melt.groupby("Bin")["Accuracy"].mean().values
plt.plot(range(len(mean_acc)), mean_acc, color='red', marker='s', markersize=8,
         linestyle='-', linewidth=3, label='Mean Accuracy')

# Annotate each mean point with its value
for i, acc in enumerate(mean_acc):
    plt.text(i - 0.38, acc + 0.02, f"{acc:.2f}", color='red', fontsize=24, 
    ha='left', fontweight='bold')

# Add chance level line
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Chance = 0.5')

# Aesthetics
plt.ylim(0.3, 1.0)
plt.yticks(fontsize=24)
# plt.title("Decoding Accuracy Across Time Bins")
plt.ylabel("Accuracy", fontsize=30)
plt.xlabel("Time Bins", fontsize=24)
#plt.xticks(ticks=plt.xticks()[0], labels=[''] * len(plt.xticks()[0])) 
plt.xticks(ticks=range(len(mean_acc)), labels=['bin1 vs bin2', 'bin1 vs bin3', 'bin1 vs bin4', 'bin1 vs bin5'], fontsize=24)
plt.gca().set_xlabel('')
# plt.legend(fontsize=16)
plt.tight_layout()

output_dir = r"S:\GVuilleumier\GVuilleumier\groups\jies\Spatial_Navigation\Final_data\GLM\MVPA_ACC\plots"
os.makedirs(output_dir, exist_ok=True)
base_filename = os.path.splitext(os.path.basename(csv_path))[0]
plot_filename = f"{base_filename}.png"
plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
plt.show()

### ANOVA and Post-hoc Tests
import pingouin as pg

anova_results = pg.rm_anova(dv='Accuracy', within='Bin', subject='subject', data=df_melt, detailed=True)
posthoc_results = pg.pairwise_ttests(dv='Accuracy', within='Bin', subject='subject', data=df_melt,
                                     padjust='fdr_bh', effsize='cohen')
anova_csv_path = os.path.join(output_dir, f"{base_filename}_ANOVA.csv")
posthoc_csv_path = os.path.join(output_dir, f"{base_filename}_posthoc.csv")

anova_results.to_csv(anova_csv_path, index=False)
posthoc_results.to_csv(posthoc_csv_path, index=False)
print("ANOVA Results:\n", anova_results)
print("\nPost-hoc Pairwise Comparisons:\n", posthoc_results)
