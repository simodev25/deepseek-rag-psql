import psycopg2
from timescale_vector import client

DATABASE_URL = "postgresql://postgres:password@localhost:5432/postgres"
conn = psycopg2.connect(DATABASE_URL)
vec_client = client.Sync(DATABASE_URL, "my_vector_taffble", 768)
vec_client.create_tables()
print("Tables créées avec succès")