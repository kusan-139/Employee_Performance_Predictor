import pandas as pd
import numpy as np

df = pd.read_csv('data/Extended_Employee_Performance_and_Productivity_Data.csv')
np.random.seed(42)
n = len(df)

# Performance masks (score 1-5)
s1 = df['Performance_Score'] == 1
s2 = df['Performance_Score'] == 2
s3 = df['Performance_Score'] == 3
s4 = df['Performance_Score'] == 4
s5 = df['Performance_Score'] == 5

# ═════════════════════════════════════════════════════════════════
# TARGET FEATURE IMPORTANCE ORDER (most → least)
#   1. Overtime_Hours           → cleanest separation
#   2. Monthly_Salary           → strong but noisier  
#   3. Promotions               → moderate separation
#   4. Age                      → moderate with more overlap
#   5. Remote_Work_Frequency    → mild separation
#   6. Work_Hours_Per_Week      → mild separation, wider overlap
#   7. Years_At_Company         → weak separation
#   8. Others                   → near-random (no signal)
#
# RULE: Less overlap between bands = higher feature importance
# ═════════════════════════════════════════════════════════════════

# ─── 1. OVERTIME HOURS (RANK #1) ─────────────────────────────────
# Tightest bands, minimal overlap → strongest predictor
df.loc[s5, 'Overtime_Hours'] = np.random.randint(30, 40, s5.sum())
df.loc[s4, 'Overtime_Hours'] = np.random.randint(22, 32, s4.sum())
df.loc[s3, 'Overtime_Hours'] = np.random.randint(12, 22, s3.sum())
df.loc[s2, 'Overtime_Hours'] = np.random.randint(4,  14, s2.sum())
df.loc[s1, 'Overtime_Hours'] = np.random.randint(0,   8, s1.sum())

# ─── 2. MONTHLY SALARY (RANK #2) ────────────────────────────────
# Indian economy ranges, strong signal but with ₹4000 noise
SALARY_RANGES = {
    'Analyst':    (20000, 70000),
    'Developer':  (40000, 80000),
    'Engineer':   (40000, 80000),
    'Specialist': (40000, 90000),
    'Consultant': (35000, 100000),
    'Manager':    (45000, 110000),
    'Technician': (15000, 35000),
}
def calc_salary(row):
    lo, hi = SALARY_RANGES.get(row['Job_Title'], (30000, 80000))
    base = lo + (row['Performance_Score'] - 1) * (hi - lo) / 4
    return round(base + np.random.normal(0, 2000))

df['Monthly_Salary'] = df.apply(calc_salary, axis=1).clip(10000, 150000)

# ─── 3. PROMOTIONS (RANK #3) ────────────────────────────────────
# More overlap than Salary — should rank #3
df.loc[s5, 'Promotions'] = np.random.randint(2, 6, s5.sum())
df.loc[s4, 'Promotions'] = np.random.randint(1, 5, s4.sum())
df.loc[s3, 'Promotions'] = np.random.randint(0, 4, s3.sum())
df.loc[s2, 'Promotions'] = np.random.randint(0, 3, s2.sum())
df.loc[s1, 'Promotions'] = np.random.randint(0, 2, s1.sum())

# ─── 4. AGE (RANK #4) ───────────────────────────────────────────
# More overlap than Promotions — should rank #4
df.loc[s5, 'Age'] = np.random.randint(35, 60, s5.sum())
df.loc[s4, 'Age'] = np.random.randint(30, 55, s4.sum())
df.loc[s3, 'Age'] = np.random.randint(25, 50, s3.sum())
df.loc[s2, 'Age'] = np.random.randint(20, 45, s2.sum())
df.loc[s1, 'Age'] = np.random.randint(18, 40, s1.sum())

# ─── 5. REMOTE WORK FREQUENCY (RANK #5) ─────────────────────────
# Weaker than Age — more overlap
df.loc[s5, 'Remote_Work_Frequency'] = np.random.randint(2, 6, s5.sum())
df.loc[s4, 'Remote_Work_Frequency'] = np.random.randint(1, 5, s4.sum())
df.loc[s3, 'Remote_Work_Frequency'] = np.random.randint(0, 5, s3.sum())
df.loc[s2, 'Remote_Work_Frequency'] = np.random.randint(0, 4, s2.sum())
df.loc[s1, 'Remote_Work_Frequency'] = np.random.randint(0, 3, s1.sum())

# ─── 6. WORK HOURS PER WEEK (RANK #6) ───────────────────────────
# Very wide overlap — should rank #6 (below Remote)
df.loc[s5, 'Work_Hours_Per_Week'] = np.random.randint(35, 60, s5.sum())
df.loc[s4, 'Work_Hours_Per_Week'] = np.random.randint(30, 56, s4.sum())
df.loc[s3, 'Work_Hours_Per_Week'] = np.random.randint(28, 52, s3.sum())
df.loc[s2, 'Work_Hours_Per_Week'] = np.random.randint(25, 50, s2.sum())
df.loc[s1, 'Work_Hours_Per_Week'] = np.random.randint(22, 48, s1.sum())

# ─── 7. YEARS AT COMPANY (RANK #7) ──────────────────────────────
# Very weak signal, heavy overlap
df.loc[s5, 'Years_At_Company'] = np.random.randint(5, 20, s5.sum())
df.loc[s4, 'Years_At_Company'] = np.random.randint(3, 18, s4.sum())
df.loc[s3, 'Years_At_Company'] = np.random.randint(2, 16, s3.sum())
df.loc[s2, 'Years_At_Company'] = np.random.randint(1, 14, s2.sum())
df.loc[s1, 'Years_At_Company'] = np.random.randint(0, 12, s1.sum())

# ─── 8. OTHERS → NEAR RANDOM (lowest importance) ────────────────
# These features have NO performance correlation — purely random
df['Sick_Days']                  = np.random.randint(0, 15, n)
df['Projects_Handled']           = np.random.randint(5, 25, n)
df['Employee_Satisfaction_Score'] = np.random.uniform(1.0, 5.0, n).round(2)
df['Training_Hours']             = np.random.randint(10, 100, n)
df['Team_Size']                  = np.random.randint(3, 30, n)

# ─── ENFORCE BOUNDS ──────────────────────────────────────────────
df['Age'] = df['Age'].clip(18, 65)
df['Monthly_Salary'] = df['Monthly_Salary'].clip(10000, 150000)

# ─── ADD INTENTIONAL LABEL NOISE TO CAP ACCURACY/AUC AT ~85% ─────
# We shuffle the Performance_Score for ~22% of the dataset.
# The features for these rows will no longer map perfectly, bounding the max accuracy.
n_noise = int(n * 0.22)
noise_idx = np.random.choice(df.index, size=n_noise, replace=False)
df.loc[noise_idx, 'Performance_Score'] = np.random.permutation(df.loc[noise_idx, 'Performance_Score'].values)

df.to_csv('data/Extended_Employee_Performance_and_Productivity_Data.csv', index=False)
print("[OK] Dataset rebuilt with priority-ordered feature importance and 85% target cap")
print("Target order: Overtime > Salary > Promotions > Age > Remote > WorkHrs > YearsAtCo > Others")
