import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PUBLICATION-GRADE FORMATTING CONFIG
# ==========================================
# Force matplotlib to use academic journal standards
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.linewidth': 1.5,       # Thicker axis borders
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 600,           # 600 DPI for print-quality export
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',     # Removes excess white space around edges
})

# Use seaborn "ticks" style (removes distracting background grids)
sns.set_theme(style="ticks", color_codes=True)
palette = {'High (8-10)': '#4daf4a', 'Mid (4-7)': '#ff7f00'} # Colorblind-friendly Green/Orange

# ==========================================
# 2. DATA LOADING AND CLEANING
# ==========================================
df = pd.read_excel("larvae_data_Fully_Graded.xlsx")
survivors = df[df['Survived_30dpf'] == 1].copy()
survivors = survivors[survivors['Quality_Tier'].isin(['High (8-10)', 'Mid (4-7)', 'Middle (4-7)'])]
survivors['Quality_Tier'] = survivors['Quality_Tier'].replace({'Middle (4-7)': 'Mid (4-7)'})

# ==========================================
# 3. HIGH-RESOLUTION FIGURE GENERATION
# ==========================================
print("Generating publication-ready figures at 600 DPI...")

# --- Figure 1: Peak Maximum Velocity (Vmax) ---
# Size: 3.5 inches wide (standard single-column journal width)
plt.figure(figsize=(4, 5))
sns.boxplot(x='Quality_Tier', y='V_max', data=survivors, palette=palette, showfliers=False, width=0.5, 
            boxprops=dict(edgecolor='black', linewidth=1.5, alpha=0.8),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5),
            medianprops=dict(color='black', linewidth=2))
sns.stripplot(x='Quality_Tier', y='V_max', data=survivors, color='black', alpha=0.4, jitter=True, size=4)
plt.title('Peak Swimming Velocity ($V_{max}$)', pad=15, fontweight='bold')
plt.ylabel('Velocity (mm/s)', fontweight='bold')
plt.xlabel('')
sns.despine() # Removes top and right borders
plt.savefig('Figure_1_Vmax_PubReady.png')
plt.close()

# --- Figure 2: Maximum Acceleration (Amax) ---
plt.figure(figsize=(4, 5))
sns.violinplot(x='Quality_Tier', y='A_max', data=survivors, palette=palette, inner='quartile', 
               linewidth=1.5, linecolor='black', alpha=0.8)
plt.title('Maximum Acceleration ($A_{max}$)', pad=15, fontweight='bold')
plt.ylabel('Acceleration (mm/s²)', fontweight='bold')
plt.xlabel('')
sns.despine()
plt.savefig('Figure_2_Amax_PubReady.png')
plt.close()

# --- Figure 3: Density Distribution of Burst Frequency ---
# Size: 5.5 inches wide for slightly wider half-page display
plt.figure(figsize=(5.5, 4.5))
sns.kdeplot(data=survivors, x='Bout_Freq_per_min', hue='Quality_Tier', fill=True, 
            palette=palette, common_norm=False, alpha=0.4, linewidth=2)
plt.title('Locomotor Burst Distribution', pad=15, fontweight='bold')
plt.xlabel('Burst Frequency (bouts/min)', fontweight='bold')
plt.ylabel('Density', fontweight='bold')
sns.despine()
plt.savefig('Figure_3_Burst_Density_PubReady.png')
plt.close()

# --- Figure 4: Multi-parametric Signature (1x3 Subplots) ---
# Size: 10 inches wide (standard double-column full page width)
fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))

# Common boxplot properties for uniformity
box_props = dict(edgecolor='black', linewidth=1.5, alpha=0.8)
whisker_props = dict(color='black', linewidth=1.5)
cap_props = dict(color='black', linewidth=1.5)
median_props = dict(color='black', linewidth=2)

# 4A: Vmax
sns.boxplot(ax=axes[0], x='Quality_Tier', y='V_max', data=survivors, palette=palette, showfliers=False, 
            boxprops=box_props, whiskerprops=whisker_props, capprops=cap_props, medianprops=median_props)
sns.stripplot(ax=axes[0], x='Quality_Tier', y='V_max', data=survivors, color='black', alpha=0.3, jitter=True, size=3)
axes[0].set_title('A. Maximum Velocity', fontweight='bold', pad=10)
axes[0].set_ylabel('Velocity (mm/s)', fontweight='bold')
axes[0].set_xlabel('')

# 4B: Burst Frequency
sns.boxplot(ax=axes[1], x='Quality_Tier', y='Bout_Freq_per_min', data=survivors, palette=palette, showfliers=False, 
            boxprops=box_props, whiskerprops=whisker_props, capprops=cap_props, medianprops=median_props)
sns.stripplot(ax=axes[1], x='Quality_Tier', y='Bout_Freq_per_min', data=survivors, color='black', alpha=0.3, jitter=True, size=3)
axes[1].set_title('B. Burst Frequency', fontweight='bold', pad=10)
axes[1].set_ylabel('Bouts / Minute', fontweight='bold')
axes[1].set_xlabel('')

# 4C: Thigmotaxis Index
sns.boxplot(ax=axes[2], x='Quality_Tier', y='Thigmotaxis_Index', data=survivors, palette=palette, showfliers=False, 
            boxprops=box_props, whiskerprops=whisker_props, capprops=cap_props, medianprops=median_props)
sns.stripplot(ax=axes[2], x='Quality_Tier', y='Thigmotaxis_Index', data=survivors, color='black', alpha=0.3, jitter=True, size=3)
axes[2].set_title('C. Thigmotaxis Index', fontweight='bold', pad=10)
axes[2].set_ylabel('Index Value (0 to 1)', fontweight='bold')
axes[2].set_xlabel('')

# Apply despine to all subplots
for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('Figure_4_Multiparametric_PubReady.png')
plt.close()

print("Execution complete! Publication-grade figures saved to directory.")
