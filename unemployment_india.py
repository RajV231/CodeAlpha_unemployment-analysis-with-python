import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')

# csv data load
file_path = 'Unemployment in India.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'.")
    print("\nInitial DataFrame head:")
    print(df.head())
    print("\nInitial DataFrame info:")
    print(df.info())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as this script.")
    exit()

# data cleaning and preprocessing

column_mapping = {
    'Region': 'Region',
    ' Date': 'Date',
    ' Frequency': 'Frequency',
    ' Estimated Unemployment Rate (%)': 'Estimated Unemployment Rate (%)',
    ' Estimated Employed': 'Estimated Employed',
    ' Estimated Labour Participation Rate (%)': 'Estimated Labour Participation Rate (%)'
}

df.columns = df.columns.str.strip()

try:
    df = df.rename(columns=column_mapping)
    print("\nColumns renamed successfully:")
    print(df.columns.tolist())
except KeyError as e:
    print(f"\nError: One of the expected columns was not found during renaming: {e}")
    print("Please check your CSV file's column headers and adjust the 'column_mapping' dictionary accordingly.")
    print(f"Your CSV columns are: {df.columns.tolist()}")
    exit()

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df.set_index('Date', inplace=True)

print("\nChecking for missing values after initial load and renaming:")
print(df.isnull().sum())

df.dropna(inplace=True)
print("\nMissing values after dropping rows:")
print(df.isnull().sum())

numeric_cols = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=numeric_cols, inplace=True)

df['Year'] = df.index.year
df['Month'] = df.index.month
df['Month_Name'] = df.index.strftime('%b')

df['Region'] = df['Region'].str.strip()

print("\nData after cleaning and preprocessing:")
print(df.head())
print(df.info())
print(f"\nUnique regions in the dataset: {df['Region'].nunique()}")


# Exploratory Data Analysis
print("\n--- Performing Exploratory Data Analysis ---")

plt.figure(figsize=(15, 7))
national_unemployment = df.groupby(df.index.to_period('M'))['Estimated Unemployment Rate (%)'].mean()
national_unemployment.index = national_unemployment.index.to_timestamp()

sns.lineplot(x=national_unemployment.index, y=national_unemployment.values, marker='o', markersize=4)
plt.title('National Average Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Average Unemployment Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
avg_unemployment_by_region = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
sns.barplot(x=avg_unemployment_by_region.values, y=avg_unemployment_by_region.index, palette='viridis')
plt.title('Average Unemployment Rate by Region')
plt.xlabel('Average Unemployment Rate (%)')
plt.ylabel('Region')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nInteractive Plotly visualization for regional trends...")
fig = px.line(df.reset_index(), x='Date', y='Estimated Unemployment Rate (%)', color='Region',
              title='Unemployment Rate Trends by Region Over Time',
              hover_data={'Estimated Employed': True, 'Estimated Labour Participation Rate (%)': True},
              height=600)
fig.update_layout(xaxis_title='Date', yaxis_title='Unemployment Rate (%)')
fig.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Estimated Unemployment Rate (%)'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Estimated Unemployment Rates')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Impact of COVID-19 

print("\n--- Analyzing COVID-19 Impact ---")

pre_covid_end = '2020-03-24'
first_lockdown_start = '2020-03-25'
first_lockdown_peak_end = '2020-06-30'
second_wave_start = '2021-03-01' 
second_wave_peak_end = '2021-06-30'

plt.figure(figsize=(15, 7))
sns.lineplot(x=national_unemployment.index, y=national_unemployment.values, marker='o', markersize=4)

plt.axvline(pd.to_datetime(pre_covid_end), color='green', linestyle='--', label='Pre-COVID End (Mar 24, 2020)')
plt.axvline(pd.to_datetime(first_lockdown_start), color='red', linestyle='--', label='First Lockdown Start (Mar 25, 2020)')
plt.axvline(pd.to_datetime(first_lockdown_peak_end), color='orange', linestyle='--', label='First Wave Peak End (Jun 30, 2020)')
plt.axvline(pd.to_datetime(second_wave_start), color='purple', linestyle='--', label='Second Wave Start (Mar 2021)')
plt.axvline(pd.to_datetime(second_wave_peak_end), color='brown', linestyle='--', label='Second Wave Peak End (Jun 2021)')

plt.title('National Average Unemployment Rate and Key COVID-19 Milestones')
plt.xlabel('Date')
plt.ylabel('Average Unemployment Rate (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

pre_covid_avg = df.loc[df.index < pre_covid_end, 'Estimated Unemployment Rate (%)'].mean()
lockdown_avg = df.loc[(df.index >= first_lockdown_start) & (df.index <= first_lockdown_peak_end), 'Estimated Unemployment Rate (%)'].mean()
post_first_wave_avg = df.loc[(df.index > first_lockdown_peak_end) & (df.index < second_wave_start), 'Estimated Unemployment Rate (%)'].mean()
second_wave_avg = df.loc[(df.index >= second_wave_start) & (df.index <= second_wave_peak_end), 'Estimated Unemployment Rate (%)'].mean()
recent_avg = df.loc[df.index > second_wave_peak_end, 'Estimated Unemployment Rate (%)'].mean()


print(f"Average Unemployment Rate (Pre-COVID, up to Mar 24, 2020): {pre_covid_avg:.2f}%")
print(f"Average Unemployment Rate (First Lockdown Peak, Mar 25 - Jun 30, 2020): {lockdown_avg:.2f}%")
print(f"Average Unemployment Rate (Post First Wave, Jul 2020 - Feb 2021): {post_first_wave_avg:.2f}%")
print(f"Average Unemployment Rate (Second Wave Peak, Mar - Jun 2021): {second_wave_avg:.2f}%")
print(f"Average Unemployment Rate (Recent, Post Second Wave): {recent_avg:.2f}%")

plt.figure(figsize=(12, 8))
lockdown_df = df[(df.index >= first_lockdown_start) & (df.index <= first_lockdown_peak_end)]
lockdown_regional_avg = lockdown_df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
sns.barplot(x=lockdown_regional_avg.values, y=lockdown_regional_avg.index, palette='magma')
plt.title('Average Unemployment Rate by Region During First COVID-19 Lockdown Peak')
plt.xlabel('Average Unemployment Rate (%)')
plt.ylabel('Region')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# identify seasonal patterns

print("\n--- Identifying Seasonal Trends ---")

plt.figure(figsize=(10, 6))
monthly_avg_unemployment = df.groupby('Month_Name')['Estimated Unemployment Rate (%)'].mean()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg_unemployment = monthly_avg_unemployment.reindex(month_order)
sns.lineplot(x=monthly_avg_unemployment.index, y=monthly_avg_unemployment.values, marker='o', color='purple')
plt.title('Average Monthly Unemployment Rate (Seasonal Pattern)')
plt.xlabel('Month')
plt.ylabel('Average Unemployment Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

if not national_unemployment.empty:
    national_unemployment_resampled = national_unemployment.asfreq('MS')
    national_unemployment_resampled = national_unemployment_resampled.interpolate(method='linear')

    if len(national_unemployment_resampled) >= 2 * 12:
        decomposition = seasonal_decompose(national_unemployment_resampled, model='additive', period=12)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        decomposition.observed.plot(ax=ax1, title='Observed')
        decomposition.trend.plot(ax=ax2, title='Trend')
        decomposition.seasonal.plot(ax=ax3, title='Seasonal')
        decomposition.resid.plot(ax=ax4, title='Residual')
        plt.suptitle('Time Series Decomposition of National Unemployment Rate', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
    else:
        print("Not enough data points for robust seasonal decomposition (need at least 2 full years of data).")
else:
    print("National unemployment series is empty, skipping seasonal decomposition.")


# insights and policy implications

print("\n--- Key Insights and Policy Implications ---")

print("\nInsights:")
print("1. Unemployment spiked sharply during COVID-19 (Apr-May 2020), with lingering volatility post-pandemic.")
print("2. Certain states consistently show higher unemployment, indicating regional and structural disparities.")
print("3. COVID-19 and lockdowns caused severe, sudden job losses; later waves had smaller, localized effects.")
print("4. Seasonal patterns reflect agricultural cycles, festivals, and academic calendars.")
print("5. A falling Labour Participation Rate (LPR) may mask true unemployment as discouraged workers exit the workforce.")

print("\nPolicy Implications:")
print("1. Launch targeted employment and skill programs in high-unemployment regions.")
print("2. Strengthen social security and expand schemes like MGNREGA, including urban initiatives.")
print("3. Support MSMEs with easier credit, reduced regulation, and digital adoption.")
print("4. Diversify rural jobs beyond agriculture to create stable, year-round employment.")
print("5. Improve labor data collection for timely, localized policy action.")
print("6. Invest in education and reskilling for future job market needs (automation, green economy).")

print("\n--- Project Analysis Complete ---")