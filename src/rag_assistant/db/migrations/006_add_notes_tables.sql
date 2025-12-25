-- Notes tables for generated/user notes and their chunks
CREATE TABLE IF NOT EXISTS notes (
    notes_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    markdown TEXT NOT NULL,
    generated_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL,
    meta_json TEXT,
    UNIQUE(subject_id, asset_id, version),
    FOREIGN KEY(asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notes_chunks (
    notes_chunk_id TEXT PRIMARY KEY,
    notes_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    section_title TEXT,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(notes_id) REFERENCES notes(notes_id) ON DELETE CASCADE
);
