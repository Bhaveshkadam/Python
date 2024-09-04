import psycopg2

def get_db_connection():
    conn = psycopg2.connect(
        dbname="pdf_management",
        user="postgres",
        password="qwerty1201",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    return conn, cur