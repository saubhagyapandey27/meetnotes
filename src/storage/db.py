import os
import sqlite3
import time

class Database:
    """
    SQLite Database Manager for MeetNotes.
    Stores recordings and generated notes.
    """
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Save in APPDATA/MeetNotes directory
            app_data = os.path.expandvars(r"%APPDATA%")
            db_dir = os.path.join(app_data, "MeetNotes")
            os.makedirs(db_dir, exist_ok=True)
            self.db_path = os.path.join(db_dir, "meetnotes.db")
        else:
            self.db_path = db_path
            
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """
        Creates the tables if they do not exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Recordings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Notes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id INTEGER NOT NULL,
                    notes_text TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_id) REFERENCES recordings (id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def add_recording(self, filename: str, path: str, duration_sec: float, source: str) -> int:
        """
        Inserts a new recording and returns its ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO recordings (filename, path, duration_sec, source) VALUES (?, ?, ?, ?)",
                (filename, path, duration_sec, source)
            )
            conn.commit()
            return cursor.lastrowid

    def add_notes(self, recording_id: int, notes_text: str, chunk_count: int) -> int:
        """
        Inserts notes for a recording and returns its ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO notes (recording_id, notes_text, chunk_count) VALUES (?, ?, ?)",
                (recording_id, notes_text, chunk_count)
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_recordings(self) -> list[dict]:
        """
        Returns all recordings joined with notes if generated.
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.id, r.filename, r.path, r.duration_sec, r.source, r.created_at, n.notes_text
                FROM recordings r
                LEFT JOIN notes n ON r.id = n.recording_id
                ORDER BY r.created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_recording_by_id(self, rec_id: int) -> dict:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_notes_for_recording(self, recording_id: int) -> dict:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM notes WHERE recording_id = ?", (recording_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_recording(self, recording_id: int) -> bool:
        """
        Deletes a recording by ID. Cascades to notes due to schema setup,
        but we also delete the physical WAV file if it exists.
        """
        rec = self.get_recording_by_id(recording_id)
        if not rec:
            return False
            
        wav_path = rec["path"]
        
        # Delete entry from db
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")  # Ensure cascade deletion works
            cursor.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))
            conn.commit()
            
        # Delete WAV file physically
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            print(f"Warning: Failed to delete physical audio file: {e}")
            
        return True
