"""
=============================================================
BONE CANCER: Clinical ML Analysis
TCGA Real Patient Data — 422 Cases
=============================================================
Author: Rosemary Kamau
ORCiD:  0009-0001-3303-2573
DOI:    (to be assigned via Zenodo)

Research Question:
    Can we predict mortality outcome in bone cancer patients
    from age, disease subtype, gender, and follow-up time?

Data Source:
    The Cancer Genome Atlas (TCGA) via GDC API
    422 real bone cancer cases

Pipeline:
    1. Data Cleaning & Preprocessing
    2. Exploratory Data Analysis
    3. Feature Engineering
    4. ML Models — Mortality Prediction
    5. Survival Analysis
    6. Key Clinical Insights
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Plot styling
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'monospace',
})

BONE_COLOR  = '#f0883e'   # orange — bone
DEAD_COLOR  = '#f85149'   # red — mortality
ALIVE_COLOR = '#56d364'   # green — survival
NEUTRAL     = '#388bfd'   # blue

print("=" * 60)
print("  BONE CANCER — TCGA Clinical ML Analysis")
print("  Author: Rosemary Kamau | ORCiD: 0009-0001-3303-2573")
print("=" * 60)


# ─────────────────────────────────────────────
# SECTION 1: DATA CLEANING
# ─────────────────────────────────────────────
"""
LEARNING NOTE — DATA CLEANING IN CLINICAL DATA:

Real clinical data has several types of messiness:
1. Missing values — patient data wasn't recorded
2. Inconsistent categories — "Not Reported", "Unknown", None
3. Age stored in DAYS not years (TCGA stores age_at_diagnosis in days!)
4. Survival time needs to be CONSTRUCTED from multiple columns

In TCGA:
- If patient is DEAD → survival time = days_to_death
- If patient is ALIVE → survival time = days_to_last_follow_up
  (we know they survived AT LEAST that long — called censored data)

This is fundamental survival analysis concept.
Censored = we lost track of them, but they were alive last we knew.
"""

print("\n📊 Loading and cleaning TCGA bone cancer data...")
df = pd.read_csv('data/tcga_bone_cancer.csv')
print(f"Raw data shape: {df.shape}")

# ── Step 1: Remove unusable rows ──
# Remove 'Not Reported' and 'Unknown' vital status — we can't use these
df = df[df['vital_status'].isin(['Alive', 'Dead'])]
print(f"After removing unknown vital status: {df.shape[0]} cases")

# ── Step 2: Convert age from days to years ──
"""
LEARNING NOTE:
TCGA stores age_at_diagnosis in DAYS from birth.
Why? Precision and privacy — exact birthdate is protected.
We convert to years for interpretability.
"""
df['age_years'] = (df['age_at_diagnosis'] / 365.25).round(1)
df['age_at_index_years'] = (df['age_at_index'] / 365.25).round(1)

# ── Step 3: Construct survival time ──
"""
LEARNING NOTE — SURVIVAL TIME CONSTRUCTION:
This is one of the most important steps in clinical ML.

Dead patients: we know exactly when they died → days_to_death
Alive patients: we know they survived at least X days → days_to_last_follow_up

We combine these into ONE survival time column.
The 'outcome' column tells us if death was observed (1) or censored (0).
"""
df['survival_days'] = np.where(
    df['vital_status'] == 'Dead',
    df['days_to_death'],
    df['days_to_last_follow_up']
)
df['outcome'] = (df['vital_status'] == 'Dead').astype(int)

# ── Step 4: Simplify disease subtypes ──
"""
LEARNING NOTE — DISEASE SUBTYPE SIMPLIFICATION:
We have 10+ disease types. Many have very few cases.
ML models struggle with rare categories.
We group into 4 clinically meaningful groups:

Osseous = bone tumours (osteosarcoma, chondrosarcoma)
Sarcoma = soft tissue bone-related sarcomas (Ewing sarcoma)
Other = everything else
"""
def simplify_disease(d):
    if 'Osseous' in str(d) or 'Chondro' in str(d):
        return 'Osseous/Chondromatous'
    elif 'Sarcoma' in str(d) or 'Soft Tissue' in str(d):
        return 'Soft Tissue Sarcoma'
    elif 'Neuroepith' in str(d):
        return 'Neuroepitheliomatous'
    else:
        return 'Other'

df['disease_group'] = df['disease_type'].apply(simplify_disease)

# ── Step 5: Clean gender and race ──
df['gender'] = df['gender'].str.title()
df['race_clean'] = df['race'].apply(
    lambda x: x if x in ['white', 'black or african american', 'asian'] else 'other/unknown'
)

print(f"\n✅ Cleaned dataset: {df.shape[0]} usable cases")
print(f"   Alive: {(df['vital_status']=='Alive').sum()}")
print(f"   Dead:  {(df['vital_status']=='Dead').sum()}")
print(f"   Mortality rate: {df['outcome'].mean()*100:.1f}%")
print(f"   Median age at diagnosis: {df['age_years'].median():.1f} years")
print(f"   Median survival: {df['survival_days'].median():.0f} days ({df['survival_days'].median()/365.25:.1f} years)")


# ─────────────────────────────────────────────
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
"""
LEARNING NOTE — EDA FOR BONE CANCER:

Key biological questions we're asking visually:
1. Who gets bone cancer? (age distribution)
2. Which subtypes are most lethal?
3. Does age at diagnosis affect survival?
4. Are there gender differences in outcome?
5. How long do patients survive after diagnosis?

Every chart should answer a CLINICAL question, not just look pretty.
"""

print("\n" + "="*60)
print("  SECTION 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('TCGA Bone Cancer — Clinical Overview (422 Real Cases)',
             fontsize=15, fontweight='bold', color='#e6edf3', y=1.02)

# ── Plot 1: Vital Status ──
ax = axes[0, 0]
counts = df['vital_status'].value_counts()
colors_vs = [ALIVE_COLOR, DEAD_COLOR]
wedges, texts, autotexts = ax.pie(
    counts.values, labels=counts.index,
    colors=colors_vs, autopct='%1.1f%%',
    startangle=90, textprops={'color': '#e6edf3'}
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax.set_title('Vital Status Distribution', fontweight='bold')

# ── Plot 2: Age at Diagnosis ──
ax = axes[0, 1]
alive_ages = df[df['vital_status']=='Alive']['age_years'].dropna()
dead_ages  = df[df['vital_status']=='Dead']['age_years'].dropna()
ax.hist(alive_ages, bins=20, alpha=0.7, color=ALIVE_COLOR,
        label=f'Alive (n={len(alive_ages)})', density=True)
ax.hist(dead_ages, bins=20, alpha=0.7, color=DEAD_COLOR,
        label=f'Dead (n={len(dead_ages)})', density=True)
ax.axvline(alive_ages.median(), color=ALIVE_COLOR, linestyle='--', linewidth=2)
ax.axvline(dead_ages.median(), color=DEAD_COLOR, linestyle='--', linewidth=2)
ax.set_title('Age at Diagnosis by Outcome')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Plot 3: Disease Group Mortality ──
ax = axes[0, 2]
disease_mortality = df.groupby('disease_group')['outcome'].agg(['mean', 'count']).reset_index()
disease_mortality['mean'] *= 100
disease_mortality = disease_mortality.sort_values('mean', ascending=True)
bars = ax.barh(disease_mortality['disease_group'],
               disease_mortality['mean'],
               color=BONE_COLOR, alpha=0.85, edgecolor='#30363d')
for bar, (_, row) in zip(bars, disease_mortality.iterrows()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{row['mean']:.1f}% (n={int(row['count'])})",
            va='center', fontsize=8, color='#e6edf3')
ax.set_title('Mortality Rate by Disease Group')
ax.set_xlabel('Mortality Rate (%)')
ax.set_xlim(0, 75)
ax.grid(axis='x', alpha=0.3)

# ── Plot 4: Survival Time Distribution ──
ax = axes[1, 0]
survival_alive = df[df['vital_status']=='Alive']['survival_days'].dropna() / 365.25
survival_dead  = df[df['vital_status']=='Dead']['survival_days'].dropna() / 365.25
ax.hist(survival_alive, bins=20, alpha=0.7, color=ALIVE_COLOR,
        label='Alive (follow-up)', density=True)
ax.hist(survival_dead, bins=20, alpha=0.7, color=DEAD_COLOR,
        label='Dead (survival)', density=True)
ax.set_title('Survival/Follow-up Time (Years)')
ax.set_xlabel('Years')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Plot 5: Gender vs Outcome ──
ax = axes[1, 1]
gender_outcome = df.groupby(['gender', 'vital_status']).size().unstack(fill_value=0)
gender_outcome_pct = gender_outcome.div(gender_outcome.sum(axis=1), axis=0) * 100
gender_outcome_pct[['Alive', 'Dead']].plot(
    kind='bar', ax=ax,
    color=[ALIVE_COLOR, DEAD_COLOR],
    edgecolor='#30363d', alpha=0.85
)
ax.set_title('Outcome by Gender (%)')
ax.set_xlabel('')
ax.set_ylabel('Percentage (%)')
ax.tick_params(axis='x', rotation=0)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# ── Plot 6: Age vs Survival Days ──
ax = axes[1, 2]
alive_plot = df[df['vital_status']=='Alive']
dead_plot  = df[df['vital_status']=='Dead']
ax.scatter(alive_plot['age_years'], alive_plot['survival_days']/365.25,
           alpha=0.5, color=ALIVE_COLOR, s=30, label='Alive')
ax.scatter(dead_plot['age_years'], dead_plot['survival_days']/365.25,
           alpha=0.5, color=DEAD_COLOR, s=30, label='Dead')
ax.set_title('Age vs Survival Time')
ax.set_xlabel('Age at Diagnosis (years)')
ax.set_ylabel('Survival/Follow-up (years)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/01_eda_overview.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("\n✅ EDA plot saved → outputs/01_eda_overview.png")


# ─────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────
"""
LEARNING NOTE — FEATURE ENGINEERING FOR BONE CANCER:

We create clinically meaningful features:

1. age_group — clinicians think in groups:
   - Paediatric (<18): osteosarcoma peak
   - Young Adult (18-40): Ewing sarcoma peak
   - Middle Age (40-60): chondrosarcoma emerging
   - Elderly (60+): chondrosarcoma dominant

2. survival_years — more interpretable than days

3. long_survival — did this patient survive >2 years?
   Why 2 years? In bone cancer, 2-year survival is a key
   clinical milestone. Surviving 2 years significantly
   predicts long-term outcomes.

4. young_onset — diagnosis under 25
   Biologically significant — young bone cancer behaves
   differently from adult bone cancer at the genomic level.
"""

print("\n" + "="*60)
print("  SECTION 3: FEATURE ENGINEERING")
print("="*60)

df['age_group'] = pd.cut(
    df['age_years'],
    bins=[0, 18, 40, 60, 100],
    labels=['Paediatric', 'Young Adult', 'Middle Age', 'Elderly']
)

df['survival_years'] = df['survival_days'] / 365.25

df['long_survival'] = (df['survival_days'] > 730).astype(int)  # >2 years

df['young_onset'] = (df['age_years'] < 25).astype(int)

df['follow_up_years'] = df['days_to_last_follow_up'] / 365.25

# Encode categoricals
le = LabelEncoder()
df['gender_enc']       = le.fit_transform(df['gender'].fillna('Unknown'))
df['disease_group_enc'] = le.fit_transform(df['disease_group'])
df['race_enc']          = le.fit_transform(df['race_clean'])
df['age_group_enc']     = le.fit_transform(df['age_group'].astype(str))

print(f"✅ Features engineered")
print(f"\nAge group distribution:")
print(df['age_group'].value_counts())
print(f"\nYoung onset (<25 years): {df['young_onset'].sum()} patients ({df['young_onset'].mean()*100:.1f}%)")
print(f"Long survivors (>2 years): {df['long_survival'].sum()} patients ({df['long_survival'].mean()*100:.1f}%)")

# ── Age group mortality chart ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bone Cancer — Clinical Feature Analysis',
             fontsize=14, fontweight='bold', color='#e6edf3')

# Age group mortality
ax = axes[0]
age_mort = df.groupby('age_group', observed=True)['outcome'].agg(['mean','count']).reset_index()
age_mort['mean'] *= 100
bars = ax.bar(age_mort['age_group'].astype(str), age_mort['mean'],
              color=BONE_COLOR, alpha=0.85, edgecolor='#30363d')
for bar, (_, row) in zip(bars, age_mort.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{row['mean']:.1f}%\n(n={int(row['count'])})",
            ha='center', va='bottom', fontsize=9, color='#e6edf3')
ax.set_title('Mortality Rate by Age Group')
ax.set_ylabel('Mortality Rate (%)')
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3)

# Disease group survival
ax = axes[1]
for group, color in zip(df['disease_group'].unique(),
                         [BONE_COLOR, ALIVE_COLOR, DEAD_COLOR, NEUTRAL]):
    subset = df[df['disease_group']==group]['survival_years'].dropna()
    ax.hist(subset, bins=15, alpha=0.6, label=f'{group} (n={len(subset)})',
            color=color, density=True)
ax.set_title('Survival Years by Disease Group')
ax.set_xlabel('Years')
ax.set_ylabel('Density')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/02_feature_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("\n✅ Feature analysis saved → outputs/02_feature_analysis.png")


# ─────────────────────────────────────────────
# SECTION 4: ML MODELS
# ─────────────────────────────────────────────
"""
LEARNING NOTE — ML FOR BONE CANCER:

We train models to predict TWO things:
1. outcome (Dead=1, Alive=0) — binary classification
2. long_survival (>2 years = 1) — binary classification

Features we use:
- age_years: continuous
- gender_enc: binary
- disease_group_enc: categorical
- young_onset: binary
- survival_years / follow_up_years: continuous

IMPORTANT CLINICAL NOTE:
We DO NOT include survival_days as a feature when predicting outcome
because that would be data leakage — we'd be using the future to
predict the future. In clinical ML this is a critical mistake.
"""

print("\n" + "="*60)
print("  SECTION 4: ML MODELS — MORTALITY PREDICTION")
print("="*60)

feature_cols = [
    'age_years', 'gender_enc', 'disease_group_enc',
    'race_enc', 'age_group_enc', 'young_onset'
]

df_ml = df[feature_cols + ['outcome', 'long_survival']].dropna()
print(f"ML dataset size: {len(df_ml)} cases after dropping NaN")
print(f"Class balance — Dead: {df_ml['outcome'].sum()}, Alive: {(df_ml['outcome']==0).sum()}")

X = df_ml[feature_cols]
y = df_ml['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100,
                                                   class_weight='balanced',
                                                   random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100,
                                                        random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n  {'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'CV Score':>12}")
print(f"  {'-'*60}")

for name, model in models.items():
    X_tr = X_train_s if name == 'Logistic Regression' else X_train
    X_te = X_test_s  if name == 'Logistic Regression' else X_test
    X_cv = X_train_s if name == 'Logistic Regression' else X_train

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring='roc_auc')

    results[name] = {
        'model': model, 'accuracy': acc, 'auc': auc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob
    }
    print(f"  {name:<25} {acc*100:>9.1f}% {auc:>10.3f}"
          f"  {cv_scores.mean():.3f}±{cv_scores.std():.3f}")

# ── ROC Curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bone Cancer — Model Performance',
             fontsize=14, fontweight='bold', color='#e6edf3')

colors_roc = ['#f0883e', '#56d364', '#79c0ff']
ax = axes[0]
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1], 'white', linestyle='--', alpha=0.4)
ax.set_title('ROC Curves — Mortality Prediction')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Feature importance
ax = axes[1]
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)
ax.barh(importances.index, importances.values, color=BONE_COLOR, alpha=0.85)
ax.set_title('Feature Importance — Random Forest')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/03_model_performance.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("\n✅ Model performance saved → outputs/03_model_performance.png")


# ─────────────────────────────────────────────
# SECTION 5: SURVIVAL ANALYSIS
# ─────────────────────────────────────────────
"""
LEARNING NOTE — KAPLAN-MEIER SURVIVAL ANALYSIS:

Kaplan-Meier answers: "What is the probability of surviving
past time T?"

For bone cancer we compare survival curves across:
- Disease subtype (which type is most lethal over time?)
- Age group (do children survive longer than adults?)

The curve drops every time a patient dies.
Censored patients (alive at last follow-up) are marked with
a tick mark on the curve — they contributed information up
to that point but we don't know what happened after.

This is why survival analysis is more powerful than just
comparing mortality rates — it accounts for WHEN death occurs
and respects censored observations.
"""

print("\n" + "="*60)
print("  SECTION 5: SURVIVAL ANALYSIS")
print("="*60)

def kaplan_meier(times, events):
    """Manual Kaplan-Meier estimator"""
    times  = np.array(times)
    events = np.array(events)
    order  = np.argsort(times)
    times  = times[order]
    events = events[order]

    unique_times = np.unique(times[events == 1])
    n = len(times)
    surv_prob = 1.0
    surv_times = [0]
    surv_probs = [1.0]

    for t in unique_times:
        at_risk = np.sum(times >= t)
        deaths  = np.sum((times == t) & (events == 1))
        surv_prob *= (1 - deaths / at_risk)
        surv_times.append(t)
        surv_probs.append(surv_prob)

    return np.array(surv_times) / 365.25, np.array(surv_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Bone Cancer — Kaplan-Meier Survival Analysis\n(TCGA Real Patient Data)',
             fontsize=14, fontweight='bold', color='#e6edf3')

# ── By Disease Group ──
ax = axes[0]
colors_km = [BONE_COLOR, ALIVE_COLOR, DEAD_COLOR, NEUTRAL]
for group, color in zip(df['disease_group'].unique(), colors_km):
    subset = df[df['disease_group'] == group].dropna(subset=['survival_days'])
    if len(subset) < 5:
        continue
    t, s = kaplan_meier(subset['survival_days'], subset['outcome'])
    ax.step(t, s * 100, color=color, linewidth=2,
            label=f"{group} (n={len(subset)})", where='post')
    ax.fill_between(t, s * 100, alpha=0.1, color=color, step='post')

ax.set_title('Survival by Disease Subtype')
ax.set_xlabel('Years from Diagnosis')
ax.set_ylabel('Survival Probability (%)')
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc='lower left')
ax.grid(alpha=0.3)
ax.axhline(50, color='white', linestyle=':', alpha=0.4)
ax.text(0.2, 51, 'Median survival', color='#8b949e', fontsize=8)

# ── By Age Group ──
ax = axes[1]
age_colors = ['#79c0ff', '#56d364', '#f0883e', '#f85149']
for age_grp, color in zip(['Paediatric', 'Young Adult', 'Middle Age', 'Elderly'],
                            age_colors):
    subset = df[df['age_group'] == age_grp].dropna(subset=['survival_days'])
    if len(subset) < 5:
        continue
    t, s = kaplan_meier(subset['survival_days'], subset['outcome'])
    ax.step(t, s * 100, color=color, linewidth=2,
            label=f"{age_grp} (n={len(subset)})", where='post')
    ax.fill_between(t, s * 100, alpha=0.1, color=color, step='post')

ax.set_title('Survival by Age Group')
ax.set_xlabel('Years from Diagnosis')
ax.set_ylabel('Survival Probability (%)')
ax.set_ylim(0, 105)
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
ax.axhline(50, color='white', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('outputs/04_survival_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✅ Survival analysis saved → outputs/04_survival_analysis.png")


# ─────────────────────────────────────────────
# SECTION 6: FINAL SUMMARY
# ─────────────────────────────────────────────

best_model = max(results.items(), key=lambda x: x[1]['auc'])

print(f"""
\n{'='*60}
  BONE CANCER ANALYSIS — KEY FINDINGS
{'='*60}

  DATASET (REAL TCGA DATA)
    Total cases:        422
    Usable cases:       {len(df)}
    Alive:              {(df['vital_status']=='Alive').sum()}
    Dead:               {(df['vital_status']=='Dead').sum()}
    Mortality rate:     {df['outcome'].mean()*100:.1f}%

  CLINICAL INSIGHTS
    Median age:         {df['age_years'].median():.1f} years
    Young onset (<25):  {df['young_onset'].sum()} patients ({df['young_onset'].mean()*100:.1f}%)
    Long survivors:     {df['long_survival'].sum()} patients ({df['long_survival'].mean()*100:.1f}%)
    Dominant type:      Osseous/Chondromatous ({(df['disease_group']=='Osseous/Chondromatous').sum()} cases)

  BEST ML MODEL
    Model:              {best_model[0]}
    ROC-AUC:            {best_model[1]['auc']:.3f}
    CV Score:           {best_model[1]['cv_mean']:.3f} ± {best_model[1]['cv_std']:.3f}

  DATA LIMITATIONS (important for publication)
    - tumor_stage:      100% missing in TCGA
    - metastasis:       {df['metastasis_at_diagnosis'].isna().sum()} missing
    - No genomic features in this clinical pull
    - Survival analysis is descriptive, not inferential
    - Small dataset limits model generalisation

{'='*60}
🧬 Analysis complete. Ready for Zenodo + ORCiD submission.
""")

df.to_csv('data/tcga_bone_cancer_cleaned.csv', index=False)
print("✅ Cleaned dataset saved → data/tcga_bone_cancer_cleaned.csv")
