import pandas as pd

# Load the bias test data
df = pd.read_csv('bias-test.csv')

print("="*50)
print("DATA QUALITY METRICS")
print("="*50)

# 1. Missingness
missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
print(f"\n1. MISSINGNESS: {missing_pct:.2f}%")
print(f"   Threshold: 0%")
print(f"   Status: {'PASS' if missing_pct == 0 else 'FAIL'}")

# 2. Bias - DPD
female_avg = df[df['Gender'] == 'Female']['AI_Score'].mean()
male_avg = df[df['Gender'] == 'Male']['AI_Score'].mean()
dpd = abs(female_avg - male_avg)

print(f"\n2. BIAS - DPD:")
print(f"   Female avg: {female_avg:.2f}")
print(f"   Male avg: {male_avg:.2f}")
print(f"   DPD: {dpd:.2f}")
print(f"   Threshold: ≤ 0.10")
print(f"   Status: {'PASS' if dpd <= 0.10 else 'FAIL'}")

# 3. Bias - DIR
lower = min(female_avg, male_avg)
higher = max(female_avg, male_avg)
dir_ratio = lower / higher

print(f"\n3. BIAS - DIR:")
print(f"   DIR: {dir_ratio:.3f}")
print(f"   Threshold: 0.80 - 1.25")
print(f"   Status: {'PASS' if 0.80 <= dir_ratio <= 1.25 else 'FAIL'}")

# 4. Variability
std = df['AI_Score'].std()
mean = df['AI_Score'].mean()
cv = std / mean

print(f"\n4. SCORE VARIABILITY:")
print(f"   Mean: {mean:.2f}")
print(f"   Std Dev: {std:.2f}")
print(f"   CV: {cv:.3f}")

# Decision
print("\n" + "="*50)
all_pass = (missing_pct == 0) and (dpd <= 0.10) and (0.80 <= dir_ratio <= 1.25)
print(f"DECISION: {'GO' if all_pass else 'NO-GO'}")
print("="*50)
