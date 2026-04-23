import joblib
import pandas as pd

pipe = joblib.load('models/employee_perf_model.pkl')
pre  = pipe.named_steps['pre']
clf  = pipe.named_steps['clf']

imp   = clf.feature_importances_
feats = pre.get_feature_names_out()

df = pd.DataFrame({
    'Feature':    [f.replace('num__', '').replace('cat__', '') for f in feats],
    'Importance': imp
}).sort_values('Importance', ascending=False)

print(df.head(12).to_string(index=False))
print()

sal_row = df[df['Feature'] == 'Monthly_Salary']
sal_pct = sal_row['Importance'].values[0] * 100 if len(sal_row) else 0
print(f"Monthly_Salary : {sal_pct:.1f}%")
print(f"All others     : {100 - sal_pct:.1f}%")
