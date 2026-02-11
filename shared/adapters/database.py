"""
Pluggable Database Adapters - Abstract base class with multiple implementations.
Swap databases by changing config without code changes.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json


class DatabaseAdapter(ABC):
    """Abstract base class for database operations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def create_document(self, document: Dict[str, Any]) -> str:
        """Create a new document record. Returns document_id."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        pass
    
    @abstractmethod
    async def list_documents(
        self, 
        status: Optional[str] = None, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List documents with optional filtering."""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document. Returns True if successful."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document. Returns True if successful."""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite implementation - great for local development."""
    
    def __init__(self, db_path: str = "./data/nnp_ai.db"):
        self.db_path = db_path
        self.connection = None
    
    async def connect(self) -> None:
        import aiosqlite
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        
        self.connection = await aiosqlite.connect(self.db_path)
        
        # Create tables if not exist
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source TEXT,
                uploaded_by TEXT,
                status TEXT,
                raw_file_path TEXT,
                extracted_data TEXT,
                signature_result TEXT,
                created_at TEXT,
                updated_at TEXT,
                thinking_traces TEXT
            )
        """)
        await self.connection.commit()
    
    async def disconnect(self) -> None:
        if self.connection:
            await self.connection.close()
    
    async def create_document(self, document: Dict[str, Any]) -> str:
        doc_id = document.get("id", str(uuid.uuid4()))
        now = datetime.utcnow().isoformat()
        
        await self.connection.execute(
            """
            INSERT INTO documents (id, source, uploaded_by, status, raw_file_path, 
                                   extracted_data, signature_result, created_at, updated_at, thinking_traces)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                document.get("source"),
                document.get("uploaded_by"),
                document.get("status", "INGESTED"),
                document.get("raw_file_path"),
                json.dumps(document.get("extracted_data", {})),
                json.dumps(document.get("signature_result", {})),
                now,
                now,
                json.dumps(document.get("thinking_traces", []))
            )
        )
        await self.connection.commit()
        return doc_id
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        async with self.connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_dict(row)
        return None
    
    async def list_documents(
        self, 
        status: Optional[str] = None, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM documents"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        # DEBUG: Log what's being stored
        print("\n" + "="*80)
        print("💾 DATABASE UPDATE")
        print("="*80)
        print(f"Document ID: {document_id}")
        print(f"Updates: {json.dumps(updates, indent=2, default=str)}")
        print("="*80 + "\n")
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ["extracted_data", "signature_result"]:
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            params.append(value)
        
        set_clauses.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(document_id)
        
        await self.connection.execute(
            f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?",
            params
        )
        await self.connection.commit()
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        await self.connection.execute(
            "DELETE FROM documents WHERE id = ?", (document_id,)
        )
        await self.connection.commit()
        return True
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        return {
            "id": row[0],
            "source": row[1],
            "uploaded_by": row[2],
            "status": row[3],
            "raw_file_path": row[4],
            "extracted_data": json.loads(row[5]) if row[5] else {},
            "signature_result": json.loads(row[6]) if row[6] else {},
            "created_at": row[7],
            "updated_at": row[8],
            "thinking_traces": json.loads(row[9]) if len(row) > 9 and row[9] else []
        }


class PostgresAdapter(DatabaseAdapter):
    """PostgreSQL implementation - for production use."""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.pool = None
    
    async def connect(self) -> None:
        import asyncpg
        self.pool = await asyncpg.create_pool(self.dsn)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source TEXT,
                    uploaded_by TEXT,
                    status TEXT,
                    raw_file_path TEXT,
                    extracted_data JSONB DEFAULT '{}',
                    signature_result JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
    
    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
    
    async def create_document(self, document: Dict[str, Any]) -> str:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (source, uploaded_by, status, raw_file_path, 
                                       extracted_data, signature_result)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                document.get("source", "manual"),
                document.get("uploaded_by", "system"),
                document.get("status", "INGESTED"),
                document.get("raw_file_path", ""),
                json.dumps(document.get("extracted_data", {})),
                json.dumps(document.get("signature_result", {}))
            )
            return str(row["id"])
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", document_id
            )
            return dict(row) if row else None
    
    async def list_documents(
        self, 
        status: Optional[str] = None, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM documents 
                    WHERE status = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                    """,
                    status, limit, offset
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM documents ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset
                )
            return [dict(row) for row in rows]
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        # Simplified - in production, build dynamic query
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents 
                SET status = COALESCE($2, status),
                    extracted_data = COALESCE($3, extracted_data),
                    signature_result = COALESCE($4, signature_result),
                    updated_at = NOW()
                WHERE id = $1
                """,
                document_id,
                updates.get("status"),
                json.dumps(updates.get("extracted_data")) if "extracted_data" in updates else None,
                json.dumps(updates.get("signature_result")) if "signature_result" in updates else None
            )
            return True
    
    async def delete_document(self, document_id: str) -> bool:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
            return True


class MongoAdapter(DatabaseAdapter):
    """MongoDB implementation - for document-heavy workloads."""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.client = None
        self.db = None
    
    async def connect(self) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self.client = AsyncIOMotorClient(self.uri)
        self.db = self.client.get_default_database()
    
    async def disconnect(self) -> None:
        if self.client:
            self.client.close()
    
    async def create_document(self, document: Dict[str, Any]) -> str:
        doc = {
            "_id": document.get("id", str(uuid.uuid4())),
            "source": document.get("source", "manual"),
            "uploaded_by": document.get("uploaded_by", "system"),
            "status": document.get("status", "INGESTED"),
            "raw_file_path": document.get("raw_file_path", ""),
            "extracted_data": document.get("extracted_data", {}),
            "signature_result": document.get("signature_result", {}),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        await self.db.documents.insert_one(doc)
        return doc["_id"]
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        doc = await self.db.documents.find_one({"_id": document_id})
        if doc:
            doc["id"] = doc.pop("_id")
        return doc
    
    async def list_documents(
        self, 
        status: Optional[str] = None, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = {"status": status} if status else {}
        cursor = self.db.documents.find(query).sort("created_at", -1).skip(offset).limit(limit)
        docs = await cursor.to_list(length=limit)
        for doc in docs:
            doc["id"] = doc.pop("_id")
        return docs
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        updates["updated_at"] = datetime.utcnow()
        await self.db.documents.update_one(
            {"_id": document_id},
            {"$set": updates}
        )
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        await self.db.documents.delete_one({"_id": document_id})
        return True


def get_database_adapter(config) -> DatabaseAdapter:
    """
    Factory function to get the appropriate database adapter based on config.
    
    Usage:
        from shared.config import get_config
        from shared.adapters.database import get_database_adapter
        
        config = get_config()
        db = get_database_adapter(config)
        await db.connect()
    """
    db_config = config.database
    
    if db_config.type == "sqlite":
        return SQLiteAdapter(db_config.sqlite.path)
    elif db_config.type == "postgres":
        pg = db_config.postgres
        return PostgresAdapter(pg.host, pg.port, pg.database, pg.user, pg.password)
    elif db_config.type == "mongo":
        return MongoAdapter(db_config.mongo.uri)
    else:
        raise ValueError(f"Unknown database type: {db_config.type}")
