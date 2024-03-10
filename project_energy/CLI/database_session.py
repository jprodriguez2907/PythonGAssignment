from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer import Typer, Option


app = Typer()

#Create database and session
DB_URI = "sqlite:///../../data/processed/database_energy.db"

engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

if __name__ == "__main__":
    app()