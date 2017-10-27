import pandas as pd
import numpy as np

age_gender = pd.read_csv("data/age_gender_bkts.csv")
# Delete 'year' column (all values are 2015)
age_gender.drop(labels='year',axis=1,inplace=True)

# Create function to compute percentage in group by
def norm_population(x):
    return x / np.sum(x)

## two group by objects. One for sum, the other for percentages

# Group by country, age tier, gender and compute sum
country_totals = age_gender.groupby(['country_destination','age_bucket','gender']).agg('sum')
# Group by country, age tier, gender and compute percent
country_pct = country_totals.groupby(level=0).apply(norm_population)
country_pct.reset_index(inplace=True)

# Split males and female for pivot operation
# We will later flatten data into one dataframe for output

male = country_pct[country_pct['gender'] == 'male']
male.name = 'males'
female = country_pct[country_pct['gender'] == 'male']
female.name = 'females'

# Put split df's by gender into list
gender_df_list = [male,female]
df_pivot_clean = []
# iterate over gender df's
for df in gender_df_list:
    # Pivot operation: Transform age_bucket rows into colums
    df_pivot = df.pivot(index = 'age_bucket', columns =  'country_destination',values='population_in_thousands')
    df_cols = df_pivot.columns
    # Rename columns to include gender prefix
    df_cols_gender = [df.name + "_" + x for x in df_cols]
    df_pivot.columns = df_cols_gender
    # Add dataframe to list
    df_pivot_clean.append(df_pivot)

# combine data frames. Concetenate over rows (by columns)
out = pd.concat(df_pivot_clean,axis=1)

out.to_csv("data/age-gender-pct-clean.csv")