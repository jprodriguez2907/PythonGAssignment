from sqlalchemy import create_engine
import pandas as pd

DB_URI = "sqlite:///../../data/processed/database_energy.db"
engine = create_engine(DB_URI, pool_pre_ping=True)
query = "SELECT * FROM cleaned_data"
data = pd.read_sql_query(query, engine)

data.head()