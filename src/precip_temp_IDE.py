### Precipitation and Temperature Initial Data Exploration

# Install pandas, matplotlib, and seaborn for plots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Change display to show all data files and confirm data displays correctly
#pd.set_option('display.max_rows',None)
#pd.set_option('display.max_columns', None)
 
datafile = pd.read_csv('../data/Precip_and_Temp.csv')
print(f'Precipitation data read to datafile successfully. ')

#Prints 6 rows of data file to confirm it correctly loaded.  
print(f'{datafile.head(3)}\n{datafile.tail(3)}')
datafile.head(3)
datafile.tail(3)
##########################################################
# Inital Data Exporation
print(f'Data description: {datafile.describe()}')
print(f'Data shape: {datafile.shape}')

#Convert 'DATE' column to datetime objects
try: 
    datafile['DATE']=pd.to_datetime(datafile['DATE'])
except Exception as e:
    print(f'Error converting "DATE" to datetime: {e}')
else: 
    print(f'"DATE" converted successfully to datetime')

#Create new columns for Year and Month 
try: 
    datafile['year']=datafile['DATE'].dt.year
except Exception as e: 
    print(f'Error creating year column: {e}')
else: 
    print(f'"year" column created successfully')
try: 
    datafile['month']=datafile['DATE'].dt.month
except Exception as e:
    print(f'Error creating "month" column: {e}')
else: print(f'"month" column created successfully')
print(datafile.head())
print(f'{datafile.isnull().sum()}')

# Impute missing precipitation values
applied_years = datafile[datafile['year'].between(1995, 1999, inclusive="both")]
applied_novembers = applied_years[applied_years['month'] == 11]
applied_novembers = applied_novembers[applied_novembers['PRCP'].isnull() == False]
datafile.iloc[6868, 3] = (datafile['PRCP'].iloc[6867]+datafile['PRCP'].iloc[6869])/2
datafile.iloc[24410,3] = (datafile['PRCP'].iloc[24409]+datafile['PRCP'].iloc[24411])/2
datafile.iloc[14093:14121, 3] = applied_novembers['PRCP'].mean()

# Impute missing maximum temperature values

# imputes November 1997, May 2017 first, since methodology is different
applied_novembers = applied_novembers[applied_novembers['TMAX'].isnull() == False]
datafile.iloc[14093:14121, 4] = applied_novembers['TMAX'].mean()
datafile.iloc[21214:21217, 4] = ((pd.to_numeric(datafile.iloc[21213, 4], errors = 'raise'))+(pd.to_numeric(datafile.iloc[21217, 4], errors='raise')))/2

# imputes rest of maximum temperatures, where fewer than 3 consecutive days were missing
to_impute_max = datafile[datafile['TMAX'].isnull() == True].index.to_list()
def impute_max(loc):
    prev = loc - 1
    subs = loc + 1
    avg = (pd.to_numeric(datafile.iloc[prev, 4], errors = 'raise') + pd.to_numeric(datafile.iloc[subs, 4], errors = 'raise'))/2
    return avg
for n in to_impute_max:
    datafile.iloc[n, 4] = impute_max(n)

# Impute missing minimum temperature values
# November 1997
applied_novembers = applied_years[applied_years['month'] == 11]
applied_novembers = applied_novembers[applied_novembers['TMIN'].isnull() == False]
datafile.iloc[14093:14121, 5] = applied_novembers['TMIN'].mean()

# April 2016 and May 2017, where each was missing 4 days
datafile.iloc[20821:20824, 5] = ((pd.to_numeric(datafile.iloc[20821, 5], errors = 'raise'))+(pd.to_numeric(datafile.iloc[20824, 5], errors='raise')))/2
datafile.iloc[21214:21217, 5] = ((pd.to_numeric(datafile.iloc[21213, 5], errors = 'raise'))+(pd.to_numeric(datafile.iloc[21217, 5], errors='raise')))/2

# imputes rest of minimum temperatures, where fewer than 3 consecutive days were missing
to_impute_min = datafile[datafile['TMIN'].isnull() == True].index.to_list()
def impute_min(loc):
    prev = loc - 1
    subs = loc + 1
    avg = (pd.to_numeric(datafile.iloc[prev, 5], errors = 'raise') + pd.to_numeric(datafile.iloc[subs, 5], errors = 'raise'))/2
    return avg

for n in to_impute_min:
    datafile.iloc[n, 5] = impute_min(n)

#group by year and sum the PRCP column
monthly_prcp=datafile.groupby(['year','month'])['PRCP'].sum().reset_index()
annual_prcp=datafile.groupby('year')['PRCP'].sum().reset_index()

#rename columns for report 
monthly_prcp.columns=['Year','Month','Total_Monthly_PRCP']
annual_prcp.columns = ['Year','Total Annual Precipation']
print(monthly_prcp)
print(annual_prcp)

yearly_summary_prcp=datafile.groupby('year').agg(
    Minimum_Value=('PRCP','min'),
    Maximum_Value=('PRCP','max'),
    Average_value=('PRCP', 'mean'), 
    Kurtosis_value=('PRCP','kurt'),
    Skewness_value=('PRCP','skew'), 
    Standard_value=('PRCP','std')   
).reset_index()
print(f'\nYearly Precipation Metrics:')
yearly_summary_prcp.rename(columns={'year':'Year'}, inplace=True)
print(yearly_summary_prcp)

#Save data to CSV file
yearly_summary_prcp.to_csv('../outputs/yearly_PRCP_summary.csv',index=False)

#Historical Annual Statistics for Precipitation
##All years 

overall_summary = datafile.agg(
    Overall_Minimum=('PRCP','min'),
    Overall_Maximum=('PRCP','max'),
    Overall_Average=('PRCP','mean'),
    Overall_Kurtosis=('PRCP','kurt'),
    Overall_Skewness=('PRCP','skew'),
    Overall_Standard_Dev=('PRCP','std')
).T

print("Overall Historical Precipitation Statistics")
print(overall_summary)
#overall_summary.to_csv('Overall_Precipitation_summary.csv',index=False)

#Save data to CSV file
overall_summary.to_csv('../outputs/Historical_Summary_Precipitation.csv',index=False)


# Distribution within each month: 
monthly_summary_prcp = datafile.groupby('month').agg(
    Minimum_Value=('PRCP','min'),
    Maximum_Value=('PRCP','max'),
    Average_value=('PRCP', 'mean'), 
    Kurtosis_value=('PRCP','kurt'),
    Skewness_value=('PRCP','skew'), 
    Standard_value=('PRCP','std')
    ).reset_index()
monthly_summary_prcp.rename(columns={'month': 'Month'}, inplace=True)
print(monthly_summary_prcp)


########################################################################################################

#Temp Data 

monthly_temp_extremes = datafile.groupby(['year','month']).agg(
    Monthly_max_temp=('TMAX','max'),
    Monthly_min_Temp=('TMIN','min')).reset_index()

print("Monthly Extremes")
print(monthly_temp_extremes.head())

yearly_temp_extremes =datafile.groupby('year').agg(
    Yearly_max_temp=('TMAX','max'),
    Yearly_min_Temp=('TMIN','min')
).reset_index()

print("Yearly Extremes")
print(yearly_temp_extremes)

yearly_summary_MaxTemp=datafile.groupby('year').agg(
    Minimum_Value=('TMAX','min'),
    Maximum_Value=('TMAX','max'),
    Average_value=('TMAX', 'mean'), 
    Kurtosis_value=('TMAX','kurt'),
    Skewness_value=('TMAX','skew'), 
    Standard_value=('TMAX','std')   
).reset_index()

print("Temp Metrics per year")
yearly_summary_MaxTemp.rename(columns={'year':'Year'}, inplace=True)
print(yearly_summary_MaxTemp)

#Save data to CSV file
yearly_summary_MaxTemp.to_csv('../outputs/yearly_MaxTemp_summary.csv',index=False)

#Historical Annual Statistics for MaxTemp
##All years 

overall_summary = datafile.agg(
    Overall_Minimum=('TMAX','min'),
    Overall_Maximum=('TMAX','max'),
    Overall_Average=('TMAX','mean'),
    Overall_Kurtosis=('TMAX','kurt'),
    Overall_Skewness=('TMAX','skew'),
    Overall_Standard_Dev=('TMAX','std')
).T
print("Overall Historical TMAX Statistics")
print(overall_summary)
overall_summary.to_csv('../outputs/Overall_TMAX_summary.csv',index=False)

#Save data to CSV file
overall_summary.to_csv('../outputs/Historical_Summary_TMAX.csv',index=False)


#################################################################################################
print('TMIN Per Year Metrics')
yearly_summary_MinTemp=datafile.groupby('year').agg(
    Minimum_Value=('TMIN','min'),
    Maximum_Value=('TMIN','max'),
    Average_value=('TMIN', 'mean'), 
    Kurtosis_value=('TMIN','kurt'),
    Skewness_value=('TMIN','skew'), 
    Standard_value=('TMIN','std')   
).reset_index()

yearly_summary_MinTemp.rename(columns={'year':'Year'}, inplace=True)
print(yearly_summary_MaxTemp)

#Save data to CSV file
yearly_summary_MinTemp.to_csv('../outputs/yearly_MinTemp_summary.csv',index=False)

#Historical Annual Statistics for MinTemp
##All years 
print("Historical Annual Statistics for MinTemp")
overall_summary = datafile.agg(
    Overall_Minimum=('TMIN','min'),
    Overall_Maximum=('TMIN','max'),
    Overall_Average=('TMIN','mean'),
    Overall_Kurtosis=('TMIN','kurt'),
    Overall_Skewness=('TMIN','skew'),
    Overall_Standard_Dev=('TMIN','std')
).T
print("Overall Historical TMIN Statistics")
print(overall_summary)
overall_summary.to_csv('../outputs/Overall_TMIN_summary.csv',index=False)

#Save data to CSV file
overall_summary.to_csv('../outputs/Historical_Summary_TMIN.csv',index=False)

###########################################################################
#Aggregated by month data all years

monthly_stats_agg= datafile.groupby([datafile['DATE'].dt.year,datafile['DATE'].dt.month]).agg({
    'PRCP':'sum',
    'TMAX':'mean',
    'TMIN':'mean'
})
print(monthly_stats_agg)

##########################################################################
Monthly_averages_agg =datafile.groupby(datafile['DATE'].dt.month).agg({
    'DATE':'count',
    'PRCP':'sum',
    'TMAX':'mean',
    'TMIN':'mean'
})
print(Monthly_averages_agg)

# ============================================================
# Boxplots: TMAX, TMIN, and PRCP by Month
# ============================================================

# Create month names
datafile['DATE'] = pd.to_datetime(datafile['DATE'])
datafile['Month'] = datafile['DATE'].dt.month_name()

month_order = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

datafile['Month'] = pd.Categorical(
    datafile['Month'],
    categories=month_order,
    ordered=True
)


# ------------------------------------------------------------
# TMAX Boxplot
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=datafile,
    x='Month',
    y='TMAX',
    palette='coolwarm'
)

plt.title('Daily Max Temperature Distribution by Month')
plt.ylabel('Temperature (°F)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../Figures/TMAX_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


# ------------------------------------------------------------
# TMIN Boxplot
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=datafile,
    x='Month',
    y='TMIN',
    palette='coolwarm'
)

plt.title('Daily Min Temperature Distribution by Month')
plt.ylabel('Temperature (°F)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../Figures/TMIN_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


# ------------------------------------------------------------
# PRCP Boxplot
# ------------------------------------------------------------

datafile['DATE']=pd.to_datetime(datafile['DATE'])
datafile['Month']=datafile['DATE'].dt.month_name()

month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']

datafile['Month']=pd.Categorical(datafile['Month'], categories=month_order, ordered=True)

rainy_days = datafile[datafile['PRCP'] > 0].copy()

plt.figure(figsize=(14, 7))

sns.boxplot(
    data=rainy_days, 
    x='Month', 
    y='PRCP', 
    palette='Blues',
    whis=1.5,
    fliersize=4
)

plt.yscale('log')

plt.title('Daily Precipitation Distribution by Month (Rainy Days Only)', fontsize=14)
plt.ylabel('Precipitation (inches) - Log Scale')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(axis='y', which='both', linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig('../Figures/PRCP_boxplot.png', dpi=300, bbox_inches='tight')

plt.close()