import sqlite3
import os

# sqlite3.connect() opens (or creates) the database file at the given path
# If the file doesn't exist yet, SQLite creates it automatically
conn = sqlite3.connect("SalesDB/sales.db")

# cursor is used to execute SQL statements on the connection
cursor = conn.cursor()

# CREATE TABLE IF NOT EXISTS — safe to run multiple times; won't overwrite existing data
# AUTOINCREMENT on PRIMARY KEY means SQLite auto-assigns id = 1, 2, 3, ...
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT    NOT NULL,
    product_name  TEXT    NOT NULL,
    quantity      INTEGER NOT NULL,
    price         REAL    NOT NULL,
    total         REAL    NOT NULL
)
""")

# INSERT INTO with multiple rows at once — more efficient than one INSERT per row
# Parentheses per row: (customer, product, qty, price, total)
cursor.execute("""
INSERT INTO orders (customer_name, product_name, quantity, price, total) VALUES
    ("John Doe",    "Laptop",     1, 1000.00, 1000.00),
    ("Jane Smith",  "Smartphone", 2,  500.00, 1000.00),
    ("Bob Johnson", "Tablet",     3,  200.00,  600.00)
""")

# conn.commit() flushes the transaction to disk — without this the INSERT is lost
conn.commit()

# conn.close() releases the file lock — always close when done
conn.close()

print("Database initialised at SalesDB/sales.db")
