###Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency


###Setting the working directory mac
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

###Importing the dataset
raw_dataset = pd.read_csv('United_States_Wind_Turbine_Database.csv')
print(raw_dataset.head())
print(raw_dataset.info())
print(raw_dataset.shape)

###Checking for missing values
print(raw_dataset.isnull().sum())

###Data cleaning

#drop the t_img_date column
dataset = raw_dataset.drop(['t_img_date'], axis=1)
#checking to make sure the column was dropped and no null values
print(dataset.isnull().sum())

#Changing the column names where every 't_' is replaced with 'turbine_'
dataset.columns = dataset.columns.str.replace('t_', 'turbine_')
#Changing the column names where every 'p_' is replaced with 'project_'
dataset.columns = dataset.columns.str.replace('p_', 'project_')
#Changing the column name of project_cap to project_capacity
dataset.rename(columns={'project_cap':'project_capacity'}, inplace=True)
#Changing the column name of turbine_cap to turbine_capacity
dataset.rename(columns={'turbine_cap':'turbine_capacity'}, inplace=True)
#Changing the column name of turbine_manu to turbine_manufacturer
dataset.rename(columns={'turbine_manu':'turbine_manufacturer'}, inplace=True)
#Changing the column name of turbine_hh to turbine_hub_height
dataset.rename(columns={'turbine_hh':'turbine_hub_height'}, inplace=True)
#Changing the column name of turbine_rd to turbine_rotor_diameter
dataset.rename(columns={'turbine_rd':'turbine_rotor_diameter'}, inplace=True)
#Changing the column name of turbine_rsa to turbine_rotor_swept_area
dataset.rename(columns={'turbine_rsa':'turbine_rotor_swept_area'}, inplace=True)
#Changing the column name of turbine_ttlh to turbine_tower_total_height
dataset.rename(columns={'turbine_ttlh':'turbine_tower_total_height'}, inplace=True)

#Checking to make sure the column names were changed
print(dataset.columns)

#Initial summary statistics table
print(dataset.describe())
#Summary stats for the numerical columns of the dataset: turbine_capacity, turbine_hub_height, turbine_rotor_diameter, turbine_rotor_swept_area, turbine_tower_total_height
print(dataset[['turbine_capacity', 'turbine_hub_height', 'turbine_rotor_diameter', 'turbine_rotor_swept_area', 'turbine_tower_total_height']].describe())
#counting the number of negative values in each of the above columns
print((dataset[['turbine_capacity', 'turbine_hub_height', 'turbine_rotor_diameter', 'turbine_rotor_swept_area', 'turbine_tower_total_height']] < 0).sum())
#counting the number of zero values in each of the above columns
print((dataset[['turbine_capacity', 'turbine_hub_height', 'turbine_rotor_diameter', 'turbine_rotor_swept_area', 'turbine_tower_total_height']] == 0).sum())
#counting the unique number of negative values in each of the above columns
print((dataset[['turbine_capacity', 'turbine_hub_height', 'turbine_rotor_diameter', 'turbine_rotor_swept_area', 'turbine_tower_total_height']] < 0).nunique())

#Changing 'missing' turbine_manufacturer values to 'Unknown'
dataset['turbine_manufacturer'] = dataset['turbine_manufacturer'].replace('missing', 'Unknown')
#Changing 'missing' turbine_model values to 'Unknown'
dataset['turbine_model'] = dataset['turbine_model'].replace('missing', 'Unknown')

##Handling special cases of MISSING YEAR values based off of other information in dataset, note that excel filtering was used to identify these special cases before dropping NaN values
#Converting the project_year column values of -9999 to NaN
dataset['project_year'] = dataset['project_year'].replace(-9999, np.nan)

##The first special case deals with the Westinghouse turbine manufacturer with a turbine capacity of 600
#Filling in NaN project_year values where turbine capacity = 600 by calculating the average project year for all projects with turbine capacity = 600 rounded to the nearest year
dataset['project_year'] = dataset['project_year'].fillna(round(dataset[dataset['turbine_capacity'] == 600]['project_year'].mean()))

##The second special case deals with multiple cases of Siemens Gamesa Renewable Energy turbine manufacturer whose location is on Scioto Ridg, there are reported missing year values, however when crossexamined by filtering only for the company it becomes apparent these values should have a year value of 2020.
#Filling in NaN project_year values where turbine manufacturer = Siemens Gamesa Renewable Energy and project name contains Scioto Ridg by replacing them with 2020
dataset.loc[(dataset['turbine_manufacturer'] == 'Siemens Gamesa Renewable Energy') & (dataset['project_name'].str.contains('Scioto Ridg')), 'project_year'] = 2020

##The third special case of missing year values deals with Senivon USA Corp that has a turbine capacity of 3200, same logic that was used in the Westinghouse special case was used here
#Filling in NaN project_year values where turbine capacity = 3200 by calculating the average project year for all projects with turbine capacity = 3200 rounded to the nearest year
dataset['project_year'] = dataset['project_year'].fillna(round(dataset[dataset['turbine_capacity'] == 3200]['project_year'].mean()))

##The fourth special case of missing year values deals with the turbine manufacturer Northern Power Systems with a turbine capacity of 100
#Filling in NaN project_year values where turbine capacity = 100 by calculating the average project year for all projects with turbine capacity = 100 rounded to the nearest year
dataset['project_year'] = dataset['project_year'].fillna(round(dataset[dataset['turbine_capacity'] == 100]['project_year'].mean()))

##The final special case of missing year values deals with the turbine manufacturer Seaforth with a turbine capacity of 60.
#Filling in NaN project_year values where turbine capacity = 60 by calculating the average project year for all projects with turbine capacity = 60 rounded to the nearest year
dataset['project_year'] = dataset['project_year'].fillna(round(dataset[dataset['turbine_capacity'] == 60]['project_year'].mean()))

##Dropping all observations that have NaN values in the project_year column
dataset = dataset.dropna(subset=['project_year'])

#printing out the number of NaN values in the project_year column
print(dataset['project_year'].isnull().sum())

##Now that all NaN values in the project_year column have been handled, we can now impute the averages for the other numerical columns dealing with the turbine dimensions by grouping by project_year and imputing the mean values for each column

#Addressing missing turbine_capacity values of -9999 by replacing them with the mean turbine_capacity value for that specific project year
dataset['turbine_capacity'] = dataset['turbine_capacity'].replace(-9999, np.nan)
dataset['turbine_capacity'] = dataset.groupby('project_year')['turbine_capacity'].transform(lambda x: x.fillna(x.mean()))
#print(dataset['turbine_capacity'].isnull().sum())

#printing years with null values
print(dataset[dataset['turbine_capacity'].isnull()]['project_year'].unique())

##Addressing years that do not have reported turbine capacity values by imputing the mean turbine capacity of the following year seeing as 1981 and 1989 are the only years with NaN values.
#Filling in NaN turbine_capacity values of project_year = 1981 by calculating the average turbine_capacity for all projects with project_year = 1982
dataset.loc[(dataset['project_year'] == 1981), 'turbine_capacity'] = round(dataset[dataset['project_year'] == 1982]['turbine_capacity'].mean())
#Filling in NaN turbine_capacity values of project_year = 1989 by calculating the average turbine_capacity for all projects with project_year = 1990
dataset.loc[(dataset['project_year'] == 1989), 'turbine_capacity'] = round(dataset[dataset['project_year'] == 1990]['turbine_capacity'].mean())

#checking to see if there are any more NaN values in the turbine_capacity column
print(dataset['turbine_capacity'].isnull().sum())
#converting project_year to integer
dataset['project_year'] = dataset['project_year'].astype(int)


##Addressing missing turbine_hub_height values of -9999 by replacing them with the mean turbine_hub_height value for that specific project year
dataset['turbine_hub_height'] = dataset['turbine_hub_height'].replace(-9999, np.nan)
dataset['turbine_hub_height'] = dataset.groupby('project_year')['turbine_hub_height'].transform(lambda x: x.fillna(x.mean()))
print(dataset['turbine_hub_height'].isnull().sum())
print(dataset[dataset['turbine_hub_height'].isnull()]['project_year'].unique())
#dealing with 1981 and 1982 missing turbine_hub_height values by imputing mean turbine_hub_height of 1983
dataset.loc[(dataset['project_year'] == 1981), 'turbine_hub_height'] = round(dataset[dataset['project_year'] == 1983]['turbine_hub_height'].mean())
dataset.loc[(dataset['project_year'] == 1982), 'turbine_hub_height'] = round(dataset[dataset['project_year'] == 1983]['turbine_hub_height'].mean())
#dealing with 1987 missing turbine_hub_height values by imputing mean turbine_hub_height of 1988
dataset.loc[(dataset['project_year'] == 1987), 'turbine_hub_height'] = round(dataset[dataset['project_year'] == 1988]['turbine_hub_height'].mean())
#dealing with 1989 and 1991 missing turbine_hub_height values by imputing mean turbine_hub_height of 1990
dataset.loc[(dataset['project_year'] == 1989), 'turbine_hub_height'] = round(dataset[dataset['project_year'] == 1990]['turbine_hub_height'].mean())
dataset.loc[(dataset['project_year'] == 1991), 'turbine_hub_height'] = round(dataset[dataset['project_year'] == 1990]['turbine_hub_height'].mean())
print(dataset['turbine_hub_height'].isnull().sum())

##Addressing missing turbine_rotor_diameter values of -9999 by replacing them with the mean turbine_rotor_diameter value for that specific project year
dataset['turbine_rotor_diameter'] = dataset['turbine_rotor_diameter'].replace(-9999, np.nan)
dataset['turbine_rotor_diameter'] = dataset.groupby('project_year')['turbine_rotor_diameter'].transform(lambda x: x.fillna(x.mean()))
print(dataset['turbine_rotor_diameter'].isnull().sum())
print(dataset[dataset['turbine_rotor_diameter'].isnull()]['project_year'].unique())
#dealing with 1981 and 1982 missing turbine_rotor_diameter values by imputing mean turbine_rotor_diameter of 1983
dataset.loc[(dataset['project_year'] == 1981), 'turbine_rotor_diameter'] = round(dataset[dataset['project_year'] == 1983]['turbine_rotor_diameter'].mean())
dataset.loc[(dataset['project_year'] == 1982), 'turbine_rotor_diameter'] = round(dataset[dataset['project_year'] == 1983]['turbine_rotor_diameter'].mean())
#dealing with 1987 missing turbine_rotor_diameter values by imputing mean turbine_rotor_diameter of 1988
dataset.loc[(dataset['project_year'] == 1987), 'turbine_rotor_diameter'] = round(dataset[dataset['project_year'] == 1988]['turbine_rotor_diameter'].mean())
#dealing with 1989 missing turbine_rotor_diameter values by imputing mean turbine_rotor_diameter of 1990
dataset.loc[(dataset['project_year'] == 1989), 'turbine_rotor_diameter'] = round(dataset[dataset['project_year'] == 1990]['turbine_rotor_diameter'].mean())
print(dataset['turbine_rotor_diameter'].isnull().sum())

##Addressing missing turbine_rotor_swept_area values of -9999 by replacing them with the mean turbine_rotor_swept_area value for that specific project year
dataset['turbine_rotor_swept_area'] = dataset['turbine_rotor_swept_area'].replace(-9999, np.nan)
dataset['turbine_rotor_swept_area'] = dataset.groupby('project_year')['turbine_rotor_swept_area'].transform(lambda x: x.fillna(x.mean()))
print(dataset['turbine_rotor_swept_area'].isnull().sum())
print(dataset[dataset['turbine_rotor_swept_area'].isnull()]['project_year'].unique())
#dealing with 1981 and 1982 missing turbine_rotor_swept_area values by imputing mean turbine_rotor_swept_area of 1983
dataset.loc[(dataset['project_year'] == 1981), 'turbine_rotor_swept_area'] = round(dataset[dataset['project_year'] == 1983]['turbine_rotor_swept_area'].mean())
dataset.loc[(dataset['project_year'] == 1982), 'turbine_rotor_swept_area'] = round(dataset[dataset['project_year'] == 1983]['turbine_rotor_swept_area'].mean())
#dealing with 1987 missing turbine_rotor_swept_area values by imputing mean turbine_rotor_swept_area of 1988
dataset.loc[(dataset['project_year'] == 1987), 'turbine_rotor_swept_area'] = round(dataset[dataset['project_year'] == 1988]['turbine_rotor_swept_area'].mean())
#dealing with 1989 missing turbine_rotor_swept_area values by imputing mean turbine_rotor_swept_area of 1990
dataset.loc[(dataset['project_year'] == 1989), 'turbine_rotor_swept_area'] = round(dataset[dataset['project_year'] == 1990]['turbine_rotor_swept_area'].mean())
print(dataset['turbine_rotor_swept_area'].isnull().sum())

##Addressing missing turbine_tower_total_height values of -9999 by replacing them with the mean turbine_tower_total_height value for that specific project year
dataset['turbine_tower_total_height'] = dataset['turbine_tower_total_height'].replace(-9999, np.nan)
dataset['turbine_tower_total_height'] = dataset.groupby('project_year')['turbine_tower_total_height'].transform(lambda x: x.fillna(x.mean()))
print(dataset['turbine_tower_total_height'].isnull().sum())
print(dataset[dataset['turbine_tower_total_height'].isnull()]['project_year'].unique())
#dealing with 1981 and 1982 missing turbine_tower_total_height values by imputing mean turbine_tower_total_height of 1983
dataset.loc[(dataset['project_year'] == 1981), 'turbine_tower_total_height'] = round(dataset[dataset['project_year'] == 1983]['turbine_tower_total_height'].mean())
dataset.loc[(dataset['project_year'] == 1982), 'turbine_tower_total_height'] = round(dataset[dataset['project_year'] == 1983]['turbine_tower_total_height'].mean())
#dealing with 1987 missing turbine_tower_total_height values by imputing mean turbine_tower_total_height of 1988
dataset.loc[(dataset['project_year'] == 1987), 'turbine_tower_total_height'] = round(dataset[dataset['project_year'] == 1988]['turbine_tower_total_height'].mean())
#dealing with 1989 missing turbine_tower_total_height values by imputing mean turbine_tower_total_height of 1990
dataset.loc[(dataset['project_year'] == 1989), 'turbine_tower_total_height'] = round(dataset[dataset['project_year'] == 1990]['turbine_tower_total_height'].mean())
#dealing with 1991 missing turbine_tower_total_height values by imputing mean turbine_tower_total_height of 1992
dataset.loc[(dataset['project_year'] == 1991), 'turbine_tower_total_height'] = round(dataset[dataset['project_year'] == 1992]['turbine_tower_total_height'].mean())
print(dataset['turbine_tower_total_height'].isnull().sum())

##Addressing 'missing' faa_ors values with 'Unknown'
dataset['faa_ors'] = dataset['faa_ors'].replace('missing', 'Unknown')
##Addressing 'missing' faa_asn values with 'Unknown'
dataset['faa_asn'] = dataset['faa_asn'].replace('missing', 'Unknown')
##Addressing 'missing' usgs_pr_id values with NaN
dataset['usgs_pr_id'] = dataset['usgs_pr_id'].replace('missing', np.nan)
##Addressing 'missing' eia_id values with NaN
dataset['eia_id'] = dataset['eia_id'].replace('missing', np.nan)

##Addressing unknown project name values with 'Unknown' if the first 7 characters of the project name are 'unknown'
dataset.loc[(dataset['project_name'].str[:7] == 'unknown'), 'project_name'] = 'Unknown'

##Saving the current cleaned dataframe to a csv file with a new name at the same filepath.
#dataset.to_csv('cleaned_dataset.csv', index=False)

#Counting the number of unique values in the turbine_state column
print(dataset['turbine_state'].nunique())
print(dataset['turbine_state'].unique())

###Data Analysis
#Function that creates a column graph of the number of wind turbines in each state
def turbine_state_graph():
    turbine_state_count = dataset['turbine_state'].value_counts()
    turbine_state_count.plot(kind='bar', figsize=(20,10))
    plt.xlabel('State')
    plt.ylabel('Number of Wind Turbines')
    plt.title('Number of Wind Turbines in Each State')
    plt.show()
    plt.clf()
#turbine_state_graph()

#Function that creates a column graph of the number of wind turbines given two states as parameters
def turbine_state_graph_two_states(state1, state2):
    turbine_state_count = dataset['turbine_state'].value_counts()
    turbine_state_count = turbine_state_count.loc[[state1, state2]]
    turbine_state_count.plot(kind='bar', figsize=(20,10))
    plt.xlabel('State')
    plt.ylabel('Number of Wind Turbines')
    plt.title('Number of Wind Turbines in Each State')
    plt.show()
    plt.clf()
#turbine_state_graph_two_states('CA', 'TX')

#Function that creates a column graph of the number of unique project names in each state ordered by descending number of unique project names
def unique_project_name_graph_by_state():
    unique_project_name_count = dataset.groupby('turbine_state')['project_name'].nunique().sort_values(ascending=False)
    unique_project_name_count.plot(kind='bar', figsize=(20,10))
    plt.xlabel('State')
    plt.ylabel('Number of Unique Project Names')
    plt.title('Number of Unique Project Names in Each State')
    plt.show()
    plt.clf()
#unique_project_name_graph_by_state()

#Function that creates a pie chart of the number of wind turbines in each state, with the bottom 20% being grouped into 'Other'
def turbine_state_pie_chart():
    turbine_state_count = dataset['turbine_state'].value_counts()
    turbine_state_count_other = turbine_state_count.iloc[12:].sum()
    turbine_state_count = turbine_state_count.iloc[:12]
    turbine_state_count['Other'] = turbine_state_count_other
    turbine_state_count.plot(kind='pie', figsize=(20,10), autopct='%1.1f%%', startangle=90)
    plt.title('Percentage of Wind Turbines in Each State')
    plt.show()
    plt.clf()
#turbine_state_pie_chart()

#Function that creates a scatterplot of two different column names as x and y values
def scatterplot(x, y):
    plt.scatter(dataset[x], dataset[y])
    #making the x-axis look nicer by removing '_' and capitalizing the first letter of each word
    x = x.replace('_', ' ').title()
    plt.xlabel(x)
    #making the y-axis look nicer by removing '_' and capitalizing the first letter of each word
    y = y.replace('_', ' ').title()
    plt.ylabel(y)
    plt.title(f'{x} vs {y}')
    plt.show()
    plt.clf()
#scatterplot('turbine_hub_height', 'turbine_rotor_diameter')

#Function that creates a line graph of the number of wind turbines produced each year given three states as parameters over time
def turbine_count_by_year_graph(state1, state2, state3):
    turbine_count_by_year = dataset.groupby(['project_year', 'turbine_state'])['turbine_state'].count()
    turbine_count_by_year = turbine_count_by_year.loc[:, [state1, state2, state3]]
    turbine_count_by_year = turbine_count_by_year.unstack()
    turbine_count_by_year.plot(kind='line', figsize=(20,10), marker='o')
    plt.xlabel('Year')
    plt.ylabel('Number of Wind Turbines')
    plt.title('Number of Wind Turbines Produced Each Year')
    plt.show()
    plt.clf()
#turbine_count_by_year_graph('CA', 'TX', 'IA')

#Function that creates a line graph showing how an average column value changes over time given a column name as a parameter
def average_column_value_by_year_graph(column_name):
    average_column_value_by_year = dataset.groupby('project_year')[column_name].mean()
    average_column_value_by_year.plot(kind='line', figsize=(20,10), marker='o')
    plt.xlabel('Year')
    plt.ylabel(column_name.replace('_', ' ').title())
    plt.title(f'Average {column_name.replace("_", " ").title()} Over Time')
    plt.show()
    plt.clf()
#average_column_value_by_year_graph('turbine_hub_height')

#Function that calculates the correlation coefficient between two columns given two column names as parameters and describes the correlation between the two columns as weak, moderate, or strong
def correlation_coefficient(column1, column2):
    correlation_coefficient = round(dataset[column1].corr(dataset[column2]), 3)
    if correlation_coefficient >= 0.7 or correlation_coefficient <= -0.7:
        print(f'The correlation between {column1.replace("_", " ").title()} and {column2.replace("_", " ").title()} is strong.')
    elif correlation_coefficient >= 0.3 or correlation_coefficient <= -0.3:
        print(f'The correlation between {column1.replace("_", " ").title()} and {column2.replace("_", " ").title()} is moderate.')
    else:
        print(f'The correlation between {column1.replace("_", " ").title()} and {column2.replace("_", " ").title()} is weak.')
    print(f'The correlation coefficient is {correlation_coefficient}')
#correlation_coefficient('turbine_hub_height', 'turbine_rotor_diameter')

#Function that calculates the chi2 statistic and p-value between two categorical columns given two column names as parameters and tell us whether or not the two columns are independent of each other (null hypothesis is that the two columns are independent of each other)
def chi2_test(column1, column2):
    chi2_statistic, p_value, dof, expected = chi2_contingency(pd.crosstab(dataset[column1], dataset[column2]))
    print(f'Chi2 Statistic: {chi2_statistic}')
    print(f'P-Value: {p_value}')
    print(f'Degrees of Freedom: {dof}')
    #print(f'Expected: {expected}')
    if p_value <= 0.05:
        print(f'We reject the null hypothesis that {column1.replace("_", " ").title()} and {column2.replace("_", " ").title()} are independent of each other.')
    else:
        print(f'We fail to reject the null hypothesis that {column1.replace("_", " ").title()} and {column2.replace("_", " ").title()} are independent of each other.')
#chi2_test('turbine_manufacturer', 'turbine_state')


#print(dataset[['turbine_capacity', 'turbine_hub_height', 'turbine_rotor_diameter', 'turbine_rotor_swept_area', 'turbine_tower_total_height']].describe())