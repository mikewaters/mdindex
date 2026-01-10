"""Migration version modules.

Each module in this package represents a database migration.
Modules must define:
    VERSION: int - The version number (must be unique and sequential)
    DESCRIPTION: str - Human-readable description
    up(conn): Function that applies the migration

Example migration (0002_add_feature.py):
    VERSION = 2
    DESCRIPTION = "Add feature table"

    def up(conn):
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS feature (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
        ''')
"""
