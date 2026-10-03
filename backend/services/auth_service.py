import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# Path to database file
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "users.db"))


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with dictionary rows."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Initializes users and sessions tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Active sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes password using NIST-recommended PBKDF2-HMAC-SHA256 with 100,000 iterations
    and a unique 16-byte cryptographic salt to prevent rainbow-table attacks.
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        bytes.fromhex(salt),
        100000
    )
    return key.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plain password against the stored PBKDF2 hash using constant-time comparison."""
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, stored_hash)


def register_user(email: str, username: str, password: str) -> Dict[str, Any]:
    """
    Registers a new user and generates an active auth session token.
    Validates input uniqueness and minimum password length.
    """
    email_clean = email.strip().lower()
    username_clean = username.strip()

    if not email_clean or "@" not in email_clean:
        raise ValueError("Please provide a valid email address.")
    if len(username_clean) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")

    pwd_hash, salt = hash_password(password)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if email or username already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email_clean,))
        if cursor.fetchone():
            raise ValueError("An account with this email already exists.")

        cursor.execute("SELECT id FROM users WHERE username = ?", (username_clean,))
        if cursor.fetchone():
            raise ValueError("This username is already taken. Please choose another.")

        # Insert user
        cursor.execute(
            "INSERT INTO users (email, username, password_hash, salt) VALUES (?, ?, ?, ?)",
            (email_clean, username_clean, pwd_hash, salt)
        )
        user_id = cursor.lastrowid

        # Generate session token
        token = secrets.token_hex(32)
        cursor.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))

        conn.commit()

        return {
            "token": token,
            "user": {
                "id": user_id,
                "email": email_clean,
                "username": username_clean,
                "created_at": datetime.now().isoformat()
            }
        }
    finally:
        conn.close()


def login_user(identifier: str, password: str) -> Dict[str, Any]:
    """
    Authenticates a user using email or username.
    Returns session token on success.
    """
    clean_id = identifier.strip().lower()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT * FROM users WHERE lower(email) = ? OR lower(username) = ?",
            (clean_id, clean_id)
        )
        user = cursor.fetchone()

        if not user or not verify_password(password, user["password_hash"], user["salt"]):
            raise ValueError("Invalid email/username or password.")

        # Generate new session token
        token = secrets.token_hex(32)
        cursor.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user["id"]))
        conn.commit()

        return {
            "token": token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "username": user["username"],
                "created_at": user["created_at"]
            }
        }
    finally:
        conn.close()


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Retrieves current user details from an active session token."""
    if not token:
        return None

    # Handle 'Bearer <token>' prefix if sent
    if token.startswith("Bearer "):
        token = token[7:].strip()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT u.id, u.email, u.username, u.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def logout_user(token: str) -> bool:
    """Invalidates and deletes a session token."""
    if token.startswith("Bearer "):
        token = token[7:].strip()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# Auto-initialize DB tables upon import
init_auth_db()
