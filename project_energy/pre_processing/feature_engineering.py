from database_session import SessionLocal
from sqlalchemy.sql import text
import pandas as pd


#Load cleaned data from SQLite database into Python object
query = text("SELECT * FROM cleaned_data")

with SessionLocal() as session:
    df_from_database = pd.DataFrame(session.execute(query))


#Drop columns for which all values are equal to zero
final_data = df_from_database.drop(columns=["generation fossil coal-derived gas", "generation fossil oil shale", "generation fossil peat", "generation geothermal", "generation marine", "generation wind offshore"])


#Change column time to datetime again
final_data['time'] = pd.to_datetime(final_data['time'], utc=True)



#Create feature for the season

# Extract month from the datetime column
final_data['month'] = final_data['time'].dt.month

# Initialize the season columns to 0
final_data['winter'] = 0
final_data['spring'] = 0
final_data['summer'] = 0
final_data['autumn'] = 0

# Assign 1 to the appropriate season column based on the month
final_data.loc[final_data['month'].isin([1, 2, 3]), 'winter'] = 1
final_data.loc[final_data['month'].isin([4, 5, 6]), 'spring'] = 1
final_data.loc[final_data['month'].isin([7, 8, 9]), 'summer'] = 1
final_data.loc[final_data['month'].isin([10, 11, 12]), 'autumn'] = 1

#Drop the 'month' column as no longer needed
final_data.drop('month', axis=1, inplace=True)



#Downsample data to daily instead of hourly
final_data['time'] = pd.to_datetime(final_data['time'])

# Convert the 'datetime' column to daily periods
final_data['date'] = final_data['time'].dt.to_period('D').dt.start_time

# Group by the new 'date' column, calculate the mean for daily values, and reset the index
final_data = final_data.groupby('date').mean().reset_index(drop=False)

#Drop column time
final_data = final_data.drop(columns='time')

#Set new date column as index
final_data.set_index('date', inplace=True)



#Create lags for the past 12 days
final_data["price(t-1)"] = final_data["price actual"].shift(1)
final_data["price(t-2)"] = final_data["price actual"].shift(2)
final_data["price(t-3)"] = final_data["price actual"].shift(3)
final_data["price(t-4)"] = final_data["price actual"].shift(4)
final_data["price(t-5)"] = final_data["price actual"].shift(5)
final_data["price(t-6)"] = final_data["price actual"].shift(6)
final_data["price(t-7)"] = final_data["price actual"].shift(7)
final_data["price(t-8)"] = final_data["price actual"].shift(8)
final_data["price(t-9)"] = final_data["price actual"].shift(9)
final_data["price(t-10)"] = final_data["price actual"].shift(10)
final_data["price(t-11)"] = final_data["price actual"].shift(11)
final_data["price(t-12)"] = final_data["price actual"].shift(12)


#Drop first 12 rows (as not all lags are available)
indices_to_drop = final_data.index[:12]
final_data = final_data.drop(indices_to_drop)


#Save final features into another table in SQLite database
with SessionLocal() as session:
    final_data.to_sql("final_data", session.get_bind(), if_exists="replace", index=False)