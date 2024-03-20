from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer import Typer
import os

app = Typer()

# Create database and session
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
processed_path = os.path.join(
    grandparent_dir, "data", "processed", "database_energy.db"
)
db_path = os.path.join(processed_path, "database_energy.db")
db_path = processed_path.replace("\\", "/")

DB_URI = f"sqlite:///{db_path}"

engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# if __name__ == "__main__":
#   app()
