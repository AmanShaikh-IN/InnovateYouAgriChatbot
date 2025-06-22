import pandas as pd
import sqlite3

csv_file = "Final_State_Specific_Farmer_Schemes.csv"  
df = pd.read_csv(csv_file)

print(df.shape)

conn = sqlite3.connect("schemes.db")
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS schemes_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Scheme Name TEXT,
        Objective TEXT,
        Key Benefits TEXT,
        Eligibility TEXT,
        Implementing Agency TEXT,
        Source TEXT,
        State TEXT,
        All_Documents_Required TEXT
    )
''')

df.to_sql("schemes_data", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print("Data stored in SQLite database 'schemes.db'")
