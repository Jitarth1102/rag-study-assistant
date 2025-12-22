CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mime_type TEXT,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,
    meta_json TEXT,
    FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS asset_index_status (
    asset_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    updated_at REAL NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS asset_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at REAL NOT NULL,
    UNIQUE(asset_id, page_num)
);

CREATE TABLE IF NOT EXISTS asset_ocr_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    ocr_json_path TEXT NOT NULL,
    text_len INTEGER NOT NULL,
    avg_conf REAL,
    needs_caption INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE(asset_id, page_num)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    text TEXT NOT NULL,
    bbox_json TEXT,
    start_block INTEGER NOT NULL,
    end_block INTEGER NOT NULL,
    created_at REAL NOT NULL
);
