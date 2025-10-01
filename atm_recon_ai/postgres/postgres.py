# SQL command to insert data
import psycopg2

db_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'example',
    'host': 'localhost',
    'port': '5432'
}

def write_to_pg(js):
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()
    try:
        # SQL command to insert data
        schema_name = "bank"
        table_name = "transactions"
        data = (js["account_id"], '2025-10-01 12:00:00')
        insert_query = f"""INSERT INTO {schema_name}.{table_name} 
        (account_id, created_at) VALUES (%s, %s);"""

        # Execute the insert command
        cursor.execute(insert_query, data)
        connection.commit()  # Commit the transaction
        print("Data inserted successfully.")

    except Exception as e:
        print(f"Error: {e}")
        if connection:
            connection.rollback()  # Rollback in case of error

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Database connection closed.")