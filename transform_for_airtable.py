import pandas as pd

# Read the source data
df = pd.read_csv('quarter_sales.csv')

# Define chip types and their corresponding columns
# H100 is the reference unit, so H100_quantity = H100e value
chip_mappings = [
    ('A100', 'A100_quantity', 'Total_A100_in_H100e'),
    ('H100', 'H100_quantity', 'H100_quantity'),
    ('H20', 'H20_quantity', 'Total_H20_in_H100e'),
    ('GB200', 'GB200_quantity', 'Total_GB200_in_H100e'),
]

# Create rows for output
rows = []

for _, row in df.iterrows():
    quarter = row['Quarter']
    start_date = row['Start Date']
    end_date = row['End Date']
    
    for chip_type, qty_col, h100e_col in chip_mappings:
        quantity = row[qty_col]
        h100e_value = row[h100e_col]
        
        # Only include rows where there's actual quantity
        if pd.notna(quantity) and quantity > 0:
            rows.append({
                'Name': f"{quarter} - {chip_type}",
                'Company': 'Nvidia',
                'Start date': start_date,
                'End date': end_date,
                'Compute estimate in H100e (median)': h100e_value,
                '# of Units': int(quantity),
                'Source / Link': '',
                'Notes': '',
                'Power estimate (TDP in GW)': '',
                'Chip type': chip_type,
                'CI': '',
                'Last Modified By': '',
                'Last Modified': ''
            })

# Create output dataframe
output_df = pd.DataFrame(rows)

# Save to CSV
output_path = 'quarter_sales_transformed.csv'
output_df.to_csv(output_path, index=False)

print(f"Transformed {len(rows)} rows")
print("\nPreview:")
print(output_df[['Name', 'Start date', 'End date', 'Compute estimate in H100e (median)', 'Chip type']].to_string())
