import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
database_url = os.getenv("DATABASE_URL")

@cl.data_layer
def get_data_layer():
    """Configure Chainlit to use SQLAlchemy with PostgreSQL for PSX project."""
    return SQLAlchemyDataLayer(conninfo=database_url)

@cl.auth
def auth_callback(username: str, password: str) -> cl.User | None:
    """Authenticate users for PSX analysis."""
    if username == "asfi@psx.com" and password == "asfi123":
        return cl.User(identifier=username, metadata={"role": "admin", "name": "Asfi", "project": "PSX"})
    return None 