# Test if work_salary_col is being used properly
import pandas as pd

# Create test data
prof_subsets = {
    '2010': pd.DataFrame({'TotalWages': [100000, 120000, 110000]}),
    '2011': pd.DataFrame({'TotalWages': [105000, 125000, 115000]}),
}

work_salary_col = 'TotalWages'

# Try the loop
yearly_stats = []
for year_str in sorted(prof_subsets.keys()):
    df_prof = prof_subsets[year_str]
    stats = {
        'Year': year_str,
        'Count': len(df_prof),
        'Mean': df_prof[work_salary_col].mean(),
    }
    yearly_stats.append(stats)

yearly_df = pd.DataFrame(yearly_stats)
print(yearly_df)
print("\nTest passed - work_salary_col works fine!")
