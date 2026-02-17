"""
Pluggable Database Adapters - Abstract base class with multiple implementations.
Swap databases by changing config without code changes.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json


DOCUMENT_MUTABLE_FIELDS = {
    "source",
    "uploaded_by",
    "status",
    "raw_file_path",
    "extracted_data",
    "signature_result",
    "thinking_traces",
}

DOCUMENT_JSON_FIELDS = {"extracted_data", "signature_result", "thinking_traces"}


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
    async def count_documents(self, status: Optional[str] = None) -> int:
        """Count documents with optional status filter."""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document. Returns True if successful."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document. Returns True if successful."""
        pass

    @abstractmethod
    async def get_document_status_history(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get status change history for a document."""
        pass

    @abstractmethod
    async def count_document_status_history(self, document_id: str) -> int:
        """Count status history records for a document."""
        pass

    @abstractmethod
    async def get_operation(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get latest operation/audit record for a document if available."""
        pass

    @abstractmethod
    async def list_operations(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List operation/audit records with optional status filter."""
        pass

    @abstractmethod
    async def count_operations(self, status: Optional[str] = None) -> int:
        """Count operation/audit records with optional status filter."""
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
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS document_status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                changed_by TEXT,
                reason TEXT
            )
        """)
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_status_history_doc ON document_status_history(document_id)")
        await self.connection.commit()
        
        # Migration: Add thinking_traces column if it doesn't exist
        try:
            await self.connection.execute("""
                ALTER TABLE documents ADD COLUMN thinking_traces TEXT
            """)
            await self.connection.commit()
            print("✅ Migration: Added thinking_traces column")
        except Exception:
            # Column already exists, ignore
            pass
    
    async def disconnect(self) -> None:
        if self.connection:
            await self.connection.close()
    
    async def create_document(self, document: Dict[str, Any]) -> str:
        doc_id = document.get("id", str(uuid.uuid4()))
        now = datetime.utcnow().isoformat()
        initial_status = document.get("status", "INGESTED")
        
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
                initial_status,
                document.get("raw_file_path"),
                json.dumps(document.get("extracted_data", {})),
                json.dumps(document.get("signature_result", {})),
                now,
                now,
                json.dumps(document.get("thinking_traces", []))
            )
        )

        await self.connection.execute(
            """
            INSERT INTO document_status_history (
                document_id, from_status, to_status, changed_at, changed_by, reason
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_id, None, initial_status, now, document.get("uploaded_by", "system"), "document_created")
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

    async def count_documents(self, status: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM documents"
        params: List[Any] = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        async with self.connection.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        mutable_updates = dict(updates)
        changed_by = mutable_updates.pop("_audit_changed_by", "system")
        reason = mutable_updates.pop("_audit_reason", "status_update")

        current_doc = await self.get_document(document_id)
        if not current_doc:
            return False

        current_status = current_doc.get("status")
        target_status = mutable_updates.get("status", current_status)
        set_clauses = []
        params = []
        
        for key, value in mutable_updates.items():
            if key not in DOCUMENT_MUTABLE_FIELDS:
                continue
            if key in DOCUMENT_JSON_FIELDS:
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            params.append(value)

        if not set_clauses:
            return True
        
        set_clauses.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(document_id)
        
        await self.connection.execute(
            f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?",
            params
        )

        now = datetime.utcnow().isoformat()
        if target_status != current_status:
            await self.connection.execute(
                """
                INSERT INTO document_status_history (
                    document_id, from_status, to_status, changed_at, changed_by, reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, current_status, target_status, now, changed_by, reason)
            )

        await self.connection.commit()
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        await self.connection.execute(
            "DELETE FROM documents WHERE id = ?", (document_id,)
        )
        await self.connection.execute(
            "DELETE FROM document_status_history WHERE document_id = ?", (document_id,)
        )
        await self.connection.commit()
        return True

    async def get_document_status_history(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        async with self.connection.execute(
            """
            SELECT document_id, from_status, to_status, changed_at, changed_by, reason
            FROM document_status_history
            WHERE document_id = ?
            ORDER BY changed_at DESC
            LIMIT ? OFFSET ?
            """,
            (document_id, limit, offset)
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "document_id": row[0],
                "from_status": row[1],
                "to_status": row[2],
                "changed_at": row[3],
                "changed_by": row[4],
                "reason": row[5],
            }
            for row in rows
        ]

    async def count_document_status_history(self, document_id: str) -> int:
        async with self.connection.execute(
            "SELECT COUNT(*) FROM document_status_history WHERE document_id = ?",
            (document_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def get_operation(self, document_id: str) -> Optional[Dict[str, Any]]:
        async with self.connection.execute(
            """
            SELECT id, document_id, from_status, to_status, changed_at, changed_by, reason
            FROM document_status_history
            WHERE document_id = ?
            ORDER BY changed_at DESC
            LIMIT 1
            """,
            (document_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": str(row[0]),
            "document_id": row[1],
            "operation": "DOCUMENT_CREATED" if row[2] is None else "STATUS_TRANSITION",
            "from_status": row[2],
            "to_status": row[3],
            "changed_at": row[4],
            "changed_by": row[5],
            "reason": row[6],
        }

    async def list_operations(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT id, document_id, from_status, to_status, changed_at, changed_by, reason
            FROM document_status_history
        """
        params: List[Any] = []

        if status:
            query += " WHERE to_status = ?"
            params.append(status)

        query += " ORDER BY changed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": str(row[0]),
                "document_id": row[1],
                "operation": "DOCUMENT_CREATED" if row[2] is None else "STATUS_TRANSITION",
                "from_status": row[2],
                "to_status": row[3],
                "changed_at": row[4],
                "changed_by": row[5],
                "reason": row[6],
            }
            for row in rows
        ]

    async def count_operations(self, status: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM document_status_history"
        params: List[Any] = []

        if status:
            query += " WHERE to_status = ?"
            params.append(status)

        async with self.connection.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0
    
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
                    thinking_traces JSONB DEFAULT '[]',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_status_history (
                    id BIGSERIAL PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    changed_at TIMESTAMPTZ DEFAULT NOW(),
                    changed_by TEXT,
                    reason TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_status_history_doc ON document_status_history(document_id)")
            await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS thinking_traces JSONB DEFAULT '[]'")
    
    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
    
    async def create_document(self, document: Dict[str, Any]) -> str:
        async with self.pool.acquire() as conn:
            initial_status = document.get("status", "INGESTED")
            row = await conn.fetchrow(
                """
                INSERT INTO documents (source, uploaded_by, status, raw_file_path, 
                                       extracted_data, signature_result, thinking_traces)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                document.get("source", "manual"),
                document.get("uploaded_by", "system"),
                initial_status,
                document.get("raw_file_path", ""),
                json.dumps(document.get("extracted_data", {})),
                json.dumps(document.get("signature_result", {})),
                json.dumps(document.get("thinking_traces", []))
            )
            document_id = str(row["id"])

            await conn.execute(
                """
                INSERT INTO document_status_history (
                    document_id, from_status, to_status, changed_by, reason
                ) VALUES ($1, $2, $3, $4, $5)
                """,
                document_id,
                None,
                initial_status,
                document.get("uploaded_by", "system"),
                "document_created"
            )

            return document_id
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", document_id
            )
            if not row:
                return None
            doc = dict(row)
            doc["id"] = str(doc["id"])
            doc["thinking_traces"] = doc.get("thinking_traces") or []
            return doc
    
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
            documents = []
            for row in rows:
                doc = dict(row)
                doc["id"] = str(doc["id"])
                doc["thinking_traces"] = doc.get("thinking_traces") or []
                documents.append(doc)
            return documents

    async def count_documents(self, status: Optional[str] = None) -> int:
        async with self.pool.acquire() as conn:
            if status:
                count = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE status = $1", status)
            else:
                count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            return int(count or 0)
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        async with self.pool.acquire() as conn:
            existing = await conn.fetchrow("SELECT * FROM documents WHERE id = $1", document_id)
            if not existing:
                return False

            mutable_updates = dict(updates)
            changed_by = mutable_updates.pop("_audit_changed_by", "system")
            reason = mutable_updates.pop("_audit_reason", "status_update")

            current_status = existing["status"]
            target_status = mutable_updates.get("status", current_status)

            set_clauses = []
            params = [document_id]
            param_idx = 2

            for key, value in mutable_updates.items():
                if key not in DOCUMENT_MUTABLE_FIELDS:
                    continue
                if key in DOCUMENT_JSON_FIELDS:
                    set_clauses.append(f"{key} = ${param_idx}::jsonb")
                    params.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = ${param_idx}")
                    params.append(value)
                param_idx += 1

            if set_clauses:
                set_clauses.append("updated_at = NOW()")
                query = f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = $1"
                await conn.execute(query, *params)

            if target_status != current_status:
                await conn.execute(
                    """
                    INSERT INTO document_status_history (
                        document_id, from_status, to_status, changed_by, reason
                    ) VALUES ($1, $2, $3, $4, $5)
                    """,
                    document_id,
                    current_status,
                    target_status,
                    changed_by,
                    reason
                )

            return True
    
    async def delete_document(self, document_id: str) -> bool:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
            return True

    async def get_document_status_history(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_id, from_status, to_status, changed_at, changed_by, reason
                FROM document_status_history
                WHERE document_id = $1
                ORDER BY changed_at DESC
                LIMIT $2 OFFSET $3
                """,
                document_id, limit, offset
            )
            result = []
            for row in rows:
                row_dict = dict(row)
                row_dict["document_id"] = str(row_dict["document_id"])
                result.append(row_dict)
            return result

    async def count_document_status_history(self, document_id: str) -> int:
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM document_status_history WHERE document_id = $1",
                document_id
            )
            return int(count or 0)

    async def get_operation(self, document_id: str) -> Optional[Dict[str, Any]]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, document_id, from_status, to_status, changed_at, changed_by, reason
                FROM document_status_history
                WHERE document_id = $1
                ORDER BY changed_at DESC
                LIMIT 1
                """,
                document_id
            )
            if not row:
                return None
            row_dict = dict(row)
            row_dict["id"] = str(row_dict["id"])
            row_dict["document_id"] = str(row_dict["document_id"])
            row_dict["operation"] = "DOCUMENT_CREATED" if row_dict.get("from_status") is None else "STATUS_TRANSITION"
            return row_dict

    async def list_operations(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, from_status, to_status, changed_at, changed_by, reason
                    FROM document_status_history
                    WHERE to_status = $1
                    ORDER BY changed_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    status, limit, offset
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, from_status, to_status, changed_at, changed_by, reason
                    FROM document_status_history
                    ORDER BY changed_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit, offset
                )
            result = []
            for row in rows:
                row_dict = dict(row)
                row_dict["id"] = str(row_dict["id"])
                row_dict["document_id"] = str(row_dict["document_id"])
                row_dict["operation"] = "DOCUMENT_CREATED" if row_dict.get("from_status") is None else "STATUS_TRANSITION"
                result.append(row_dict)
            return result

    async def count_operations(self, status: Optional[str] = None) -> int:
        async with self.pool.acquire() as conn:
            if status:
                count = await conn.fetchval("SELECT COUNT(*) FROM document_status_history WHERE to_status = $1", status)
            else:
                count = await conn.fetchval("SELECT COUNT(*) FROM document_status_history")
            return int(count or 0)


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
        initial_status = document.get("status", "INGESTED")
        doc = {
            "_id": document.get("id", str(uuid.uuid4())),
            "source": document.get("source", "manual"),
            "uploaded_by": document.get("uploaded_by", "system"),
            "status": initial_status,
            "raw_file_path": document.get("raw_file_path", ""),
            "extracted_data": document.get("extracted_data", {}),
            "signature_result": document.get("signature_result", {}),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        await self.db.documents.insert_one(doc)

        await self.db.document_status_history.insert_one({
            "document_id": doc["_id"],
            "from_status": None,
            "to_status": initial_status,
            "changed_at": datetime.utcnow(),
            "changed_by": document.get("uploaded_by", "system"),
            "reason": "document_created"
        })

        return doc["_id"]
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        doc = await self.db.documents.find_one({"_id": document_id})
        if doc:
            doc["id"] = str(doc.pop("_id"))
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
            doc["id"] = str(doc.pop("_id"))
        return docs

    async def count_documents(self, status: Optional[str] = None) -> int:
        query = {"status": status} if status else {}
        return int(await self.db.documents.count_documents(query))
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        existing = await self.db.documents.find_one({"_id": document_id})
        if not existing:
            return False

        mutable_updates = dict(updates)
        changed_by = mutable_updates.pop("_audit_changed_by", "system")
        reason = mutable_updates.pop("_audit_reason", "status_update")

        current_status = existing.get("status")
        target_status = mutable_updates.get("status", current_status)
        filtered_updates = {k: v for k, v in mutable_updates.items() if k in DOCUMENT_MUTABLE_FIELDS}

        if not filtered_updates:
            return True

        filtered_updates["updated_at"] = datetime.utcnow()
        await self.db.documents.update_one(
            {"_id": document_id},
            {"$set": filtered_updates}
        )

        if target_status != current_status:
            await self.db.document_status_history.insert_one({
                "document_id": document_id,
                "from_status": current_status,
                "to_status": target_status,
                "changed_at": datetime.utcnow(),
                "changed_by": changed_by,
                "reason": reason
            })

        return True
    
    async def delete_document(self, document_id: str) -> bool:
        await self.db.documents.delete_one({"_id": document_id})
        await self.db.document_status_history.delete_many({"document_id": document_id})
        return True

    async def get_document_status_history(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        cursor = self.db.document_status_history.find(
            {"document_id": document_id}
        ).sort("changed_at", -1).skip(offset).limit(limit)
        rows = await cursor.to_list(length=limit)
        for row in rows:
            row["id"] = str(row.pop("_id"))
        return rows

    async def count_document_status_history(self, document_id: str) -> int:
        return int(await self.db.document_status_history.count_documents({"document_id": document_id}))

    async def get_operation(self, document_id: str) -> Optional[Dict[str, Any]]:
        tx = await self.db.document_status_history.find_one(
            {"document_id": document_id},
            sort=[("changed_at", -1)]
        )
        if not tx:
            return None
        tx["id"] = str(tx.pop("_id"))
        tx["operation"] = "DOCUMENT_CREATED" if tx.get("from_status") is None else "STATUS_TRANSITION"
        return tx

    async def list_operations(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = {"to_status": status} if status else {}
        cursor = self.db.document_status_history.find(query).sort("changed_at", -1).skip(offset).limit(limit)
        rows = await cursor.to_list(length=limit)
        for row in rows:
            row["id"] = str(row.pop("_id"))
            row["operation"] = "DOCUMENT_CREATED" if row.get("from_status") is None else "STATUS_TRANSITION"
        return rows

    async def count_operations(self, status: Optional[str] = None) -> int:
        query = {"to_status": status} if status else {}
        return int(await self.db.document_status_history.count_documents(query))


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
