import sqlite3
import os
import time

def connect_db(path):
    """
    Establishes a connection to the SQLite database and returns the connection and cursor objects.
    """
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON;")
    cursor = connection.cursor()
    return connection, cursor

def disconnect_db(connection):
    """
    Commits changes and closes the database connection.
    """
    connection.commit()
    connection.close()

def create_database(db_name="./01_data/project_database.db"):
    """
    Creates an SQLite database with tables to store PDF information.
    """
    connection, cursor = connect_db(db_name)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdfs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        size REAL,
        date TEXT,
        in_use BOOLEAN DEFAULT TRUE,
        last_use TEXT,
        checked BOOLEAN DEFAULT FALSE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id INTEGER,
        pdf_id_old INTEGER,
        date TEXT NOT NULL,
        action TEXT NOT NULL,
        FOREIGN KEY (pdf_id) REFERENCES pdfs(id) 
            ON DELETE CASCADE 
            ON UPDATE CASCADE,
        FOREIGN KEY (pdf_id_old) REFERENCES pdfs(id)
            ON DELETE CASCADE 
            ON UPDATE CASCADE,
        CHECK (pdf_id IS NOT NULL OR pdf_id_old IS NOT NULL)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        pdf_id INTEGER NOT NULL,
        page INTEGER NOT NULL,
        FOREIGN KEY (pdf_id) REFERENCES pdfs(id) 
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    """)
    
    disconnect_db(connection)
    print(f"Database '{db_name}' created successfully.")

def register_pdf(cursor, name, path, date, size):
    """
    Inserts a new PDF record into the database and logs the creation.
    Uses parameterized queries to prevent SQL injection.
    """
    cursor.execute("""INSERT INTO pdfs (name, path, size, date, in_use, checked) VALUES (?, ?, ?, ?, TRUE, TRUE)""", (name, path, size, date))
    pdf_id = cursor.lastrowid
    cursor.execute("""INSERT INTO logs (pdf_id, pdf_id_old, date, action) VALUES (?, NULL, ?, 'CREATION')""", (pdf_id, date))

def modify_pdf(cursor, name, path, size, date, pdf_old_id):
    """
    Updates the PDF record in the database and logs the modification.
    Uses parameterized queries to prevent SQL injection.
    """
    cursor.execute("""UPDATE pdfs SET in_use = FALSE, last_use = ? WHERE id = ?""", (date, pdf_old_id))
    cursor.execute("""INSERT INTO pdfs (name, path, size, date, in_use, checked) VALUES (?, ?, ?, ?, TRUE, TRUE)""", (name, path, size, date))
    pdf_id = cursor.lastrowid
    cursor.execute("""INSERT INTO logs (pdf_id, pdf_id_old, date, action) VALUES (?, ?, ?, 'MODIFICATION')""", (pdf_id, pdf_old_id, date))

def stop_using_pdf(cursor, pdf_id, date):
    """
    Marks a PDF as no longer in use and logs the action.
    """
    cursor.execute("""UPDATE pdfs SET in_use = FALSE, last_use = ? WHERE id = ?""", (date, pdf_id))
    cursor.execute("""INSERT INTO logs (pdf_id, pdf_id_old, date, action) VALUES (NULL, ?, ?, 'NO_USE')""", (pdf_id, date))
    return "NO_USE"

def detect_new_or_updates(cursor, name, path, size, date):
    """
    Detects whether the PDF file is already registered in the database.
    If the file exists but has changed, it updates the record.
    """
    cursor.execute("""SELECT id, size, date FROM pdfs WHERE path = ? AND in_use = TRUE """, (path,))
    data = cursor.fetchone()
    #print
    
    if data:
        pdf_id, stored_size, stored_date = data
        if stored_size != size or stored_date != date:
            modify_pdf(cursor, name, path, size, date, pdf_id)
            return "MODIFIED"
        cursor.execute("""UPDATE pdfs SET checked = TRUE WHERE id = ?""", (pdf_id,))
        return "NO_CHANGE"

    else:
        register_pdf(cursor, name, path, date, size)
        return "NEW"

def scan_directory(directory, pdfs=None):
    """
    Recursively scans a directory to detect PDFs and their metadata.
    Returns a list of tuples containing (name, path, size, modification_date).
    """
    if pdfs is None:
        pdfs = []
    
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            scan_directory(path, pdfs)
        else:
            info = os.stat(path)
            if name.endswith('.pdf'):
                pdfs.append((name, path, info.st_size, time.ctime(info.st_mtime)))
    
    return pdfs

def update_database(db_name="./01_data/project_database.db", directory="./01_data/BBDD_Normativa_UPV"):
    """
    Updates the database with the PDFs found in the specified directory.
    """
    connection, cursor = connect_db(db_name)
    pdfs = scan_directory(directory)

    # Put all the in use pdfs to false
    cursor.execute("UPDATE pdfs SET checked = FALSE WHERE in_use = TRUE")
    changed = 0
    for pdf in pdfs:
        name, path, size, date = pdf
        status = detect_new_or_updates(cursor, name, path, size, date)
        if status != "NO_CHANGE":
            changed += 1
            print(f"{status} - {name}")
        #print(f"{status} - {name}")

    not_in_use = cursor.execute("SELECT id, name FROM pdfs WHERE checked = FALSE AND in_use = TRUE")
    deleted = not_in_use.fetchall()
    for row in deleted:
        pdf_id, name = row
        status = stop_using_pdf(cursor, pdf_id, time.ctime())
        print(f"{status} - {name}")
    
    disconnect_db(connection)
    print("Database updated successfully.")
    if changed == 0 and len(deleted) == 0:
        print("No changes detected.")


if __name__ == "__main__":
    create_database(db_name="./01_data/project_database.db")
    update_database(db_name="./01_data/project_database.db", directory="./01_data/pdf_actuales")

    