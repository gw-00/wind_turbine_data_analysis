# wind_turbine_data_analysis
Personal portfolio project of wind turbine data in the United States.

#Original data for this project can be found at the below link.
https://atlas.eia.gov/datasets/eia::united-states-wind-turbine-database-uswtdb/about

Data was first cleaned and then functions were developed to analyze correlations between the variables.
Missing values were imputed using averages for each year, and cases of missing year values were imputed using the rounded average of the turbine capacity column if it was present as a placeholder for the technological advancement of the time period. Only a handful of observations were dropped entirely. Comments were placed throughout in order to guide. Years that had no data utilized the average data for the closest year that had data.

There is a Tableau Public Workbook for this project at the following address for additional visualizations. To note the visualizations use the .csv file that is developed with the cleaned data found later in the code.
https://public.tableau.com/app/profile/graham.ward

