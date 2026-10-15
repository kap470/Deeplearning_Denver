### Streamflow Initial Data Exploration 

## Set Up  
# install pandas, matplotlib, and seaborn for plots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
datafile = pd.read_csv('../data/streamflow.csv')
print(f'Streamflow data read to datafile successfully. ')

#Prints 6 rows of data file to confirm data displays correctly.  
print(f'{datafile.head(3)}\n{datafile.tail(3)}')
datafile.head(3)

# Calculate and print the number of rows and columns that this dataset contains
rows, columns = datafile.shape
print(f"Number of rows: {rows}\nNumber of columns: {columns}\n")

# Null Values: 
datafile.isnull().sum()

## Data Cleaning
# Convert 'time' column to datetime objects
try: 
    datafile['time']=pd.to_datetime(datafile['time'])
except Exception as e:
    print(f'Error converting "time" to datetime: {e}')
else: 
    print(f'"time" converted successfully to datetime')
print(f'datafile details: \n{datafile.describe()}')

# Create new columns for year and month
try: 
    datafile['year']=datafile['time'].dt.year
except Exception as e:
    print(f'Error creating "year" column: {e}')
else: 
    print(f'"year" column created successfully')
    
try: 
    datafile['month'] = datafile['time'].dt.month
except Exception as e: 
    print (f'Error creating "month" column: {e}')
else: 
    print(f'"month" column created successfully')
    
# Save cleaned data to streamflow_cleaned.csv
try: 
    datafile.to_csv(../data/streamflow_cleaned.csv)
except Exception as e:
    print('Error saving streamflow_cleaned.csv')
else: 
    print(f'streamflow_cleaned.csv created successfully')

# Yearly view
yearly_view=datafile.groupby('year')['value'].sum().reset_index()
print("Yearly Streamflow Totals")
yearly_view.columns=['Year','Total_value']
print(yearly_view)
print()


# Month/per year 
monthly_totals = datafile.groupby(['year','month'])['value'].sum().reset_index()
monthly_totals.columns=['Year','Month','Total_value']
print("Monthly Totals By Year")
print(monthly_totals)
print()

# Change display to show all data files and confirm data displays correctly
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)

# Save data to CSV file
monthly_totals.to_csv('../outputs/monthly_streamflow_totals_per_year.csv',index=False)

yearly_summary_streamflow=datafile.groupby(datafile['time'].dt.year).agg(
    Minimum_Value=('value','min'),
    Maximum_Value=('value','max'),
    Average_value=('value', 'mean'), 
    Kurtosis_value=('value','kurt'),
    Skewness_value=('value','skew'), 
    Standard_value=('value','std')   
).reset_index()

yearly_summary_streamflow.rename(columns={'time':'Year'}, inplace=True)

print("Yearly Summary Stat Metrics")
print(yearly_summary_streamflow)


# Save data to CSV file
yearly_summary_streamflow.to_csv('../outputs/yearly_streamflow_summary.csv',index=False)


############################################################################################
## All years 

overall_summary = datafile.agg(
    Overall_Minimum=('value','min'),
    Overall_Maximum=('value','max'),
    Overall_Average=('value','mean'),
    Overall_Kurtosis=('value','kurt'),
    Overall_Skewness=('value','skew'),
    Overall_Standard_Dev=('value','std')
).T

print("Overall Historical Statistics(All Years)")
print(overall_summary)


# Save data to CSV file
overall_summary.to_csv('../outputs/Historical_Summary_Streamflow.csv',index=False)


######################################################################################################
# Plotting data

## Streamflow plots

datafile['Month']=datafile['time'].dt.month_name()

month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']

datafile['Month']=pd.Categorical(datafile['Month'], categories=month_order, ordered=True)

plt.figure(figsize=(12,6))

sns.boxplot(data=datafile,x='Month',y='value', palette='coolwarm')

plt.title('Streamflow ft^3/s All Months')
plt.ylabel('Flow ft^3/s')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../Figures/streamflow_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
