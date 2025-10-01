CREATE SCHEMA IF NOT EXISTS [postgres].[bank];

CREATE TABLE IF NOT EXISTS [postgres].[bank].[transactions] (
    txn_id SERIAL PRIMARY KEY,
    account_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
