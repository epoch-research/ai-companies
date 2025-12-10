import pandas as pd

# Read the source data
df = pd.read_csv('nvidia_quarterly_by_company.csv')

# Create rows for output
rows = []

for _, row in df.iterrows():
    company = row['Company']
    quarter = row['Quarter']
    start_date = row['Start Date']
    end_date = row['End Date']
    h100e_median = row['h100_equivs_median']
    h100e_10th = row['h100_equivs_10th']
    h100e_90th = row['h100_equivs_90th']
    
    rows.append({
        'Name': f"{company} - {quarter}",
        'Compute owner': company,
        'Designer': 'Nvidia',
        'Start date': start_date,
        'End date': end_date,
        'Compute estimate in H100e (median)': h100e_median,
        'Source / Link': '',
        'Notes': '',
        'Power estimate (TDP in GW)': '',
        'Chip type': 'Nvidia (total)',
        'CI': f"{int(h100e_10th)}-{int(h100e_90th)}",
        'Last Modified By': '',
        'Last Modified': '',
        'Cost estimate (USD)': '',
        'Select': ''
    })

# Create output dataframe
output_df = pd.DataFrame(rows)

# Save to CSV
output_path = 'nvidia_quarterly_by_company_transformed.csv'
output_df.to_csv(output_path, index=False)

print(f"Transformed {len(rows)} rows")
print("\nPreview:")
print(output_df[['Name', 'Compute owner', 'Compute estimate in H100e (median)', 'CI', 'Chip type']].head(20).to_string())
