import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from database_session import SessionLocal

#Load data
df_energy = pd.read_csv("../../data/raw/energy_dataset.csv")
df_weather = pd.read_csv("../../data/raw/weather_features.csv")


#rename columns
df_weather = df_weather.rename({"dt_iso":"time"},axis=1)


#Change data types
df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)


#Understand range of dates
#print(df_energy['time'].min(),df_energy['time'].max())
#print(df_weather['time'].min(),df_weather['time'].max())


#Analyze duplicate values in times
number_of_duplicates_energy = df_energy.duplicated(subset='time', keep=False).sum()
number_of_duplicates_weather = df_weather.duplicated(subset='time', keep=False).sum()
#print(f"Number of duplicates energy dataset: {number_of_duplicates_energy}")
#print(f"Number of duplicates weather dataset: {number_of_duplicates_weather}")


#Keep only the rows in weather data set for Madrid (as deviations between cities are small for most variables)
df_weather_filtered = df_weather[df_weather["city_name"] == 'Madrid']


#Join the two data sets
joined_df = pd.merge(df_weather_filtered, df_energy, on="time")

#Investigate missing values
numeric_df = joined_df.select_dtypes(include=['number'])
missing_absolute = numeric_df.isnull().sum()
missing_relative = (numeric_df.isnull().sum() / len(joined_df)) * 100
missing_table = pd.DataFrame({
    'Absolute Missing Values': missing_absolute,
    'Relative Missing Values (%)': missing_relative
})
missing_table_with_values = missing_table[missing_table['Absolute Missing Values'] > 0]
#print(missing_table_with_values)


#Drop columns with 100% missing values
joined_df = joined_df.drop(columns=["forecast wind offshore eday ahead", "generation hydro pumped storage aggregated"])


#Fill missing values for columns with <10% missing values using the interpolate method
numeric_cols = joined_df.select_dtypes(include=['number']).columns
joined_df[numeric_cols] = joined_df[numeric_cols].interpolate(method='linear', axis=0, limit_direction='forward', inplace=False)


#Drop irrelevant categorical columns (info contained in numerical columns like "clouds"
joined_df = joined_df.drop(columns=["city_name", "weather_main", "weather_description", "weather_icon"])


#Replace outliers with column's mean, except target variable
cols_to_include = joined_df.columns.difference(['price actual'])
for col in cols_to_include:
    col_mean = joined_df[col].mean()
    col_std = joined_df[col].std(ddof=0)
    joined_df[col + '_zscore'] = (joined_df[col] - col_mean) / col_std
    zscore_threshold = 2.58
    # Replace outliers with the column's mean
    joined_df[col] = np.where((joined_df[col + '_zscore'] > zscore_threshold) |
                              (joined_df[col + '_zscore'] < -zscore_threshold),
                              col_mean,
                              joined_df[col])
#Drop the Z-score columns as no longer needed
joined_df = joined_df.drop(columns=[col + '_zscore' for col in cols_to_include])


#Scale all variables except time and target variable price
cols_to_scale = joined_df.columns.difference(['price actual', 'time'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(joined_df[cols_to_scale])
scaled_df = pd.DataFrame(scaled_features, columns=cols_to_scale, index=joined_df.index)
joined_df.update(scaled_df)


#Load data into SQLite database in folder "data" - "processed"
with SessionLocal() as session:
    joined_df.to_sql("cleaned_data", session.get_bind(), if_exists="replace", index=False)
