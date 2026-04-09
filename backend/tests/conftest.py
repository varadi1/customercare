"""Shared test fixtures — real PostgreSQL with transaction rollback."""
import asyncio, pytest, asyncpg

PG_DSN = "postgresql://klara:klara_docs_2026@localhost:5433/customercare"

class AsyncDB:
    def __init__(self): self._conn = self._tx = None
    def setup(self):
        async def _s():
            self._conn = await asyncpg.connect(PG_DSN)
            self._tx = self._conn.transaction(); await self._tx.start(); return self._conn
        return asyncio.get_event_loop().run_until_complete(_s())
    def teardown(self):
        async def _t():
            if self._tx: await self._tx.rollback()
            if self._conn: await self._conn.close()
        asyncio.get_event_loop().run_until_complete(_t())

def run(coro): return asyncio.get_event_loop().run_until_complete(coro)

@pytest.fixture(scope="session", autouse=True)
def _setup_event_loop():
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); yield loop; loop.close()

@pytest.fixture
def db_conn():
    db = AsyncDB(); conn = db.setup(); yield conn; db.teardown()

@pytest.fixture
def ensure_reasoning_traces_table(db_conn):
    run(db_conn.execute("""CREATE TABLE IF NOT EXISTS reasoning_traces (
        id SERIAL PRIMARY KEY, query_text TEXT NOT NULL, query_embedding vector(1024),
        email_message_id VARCHAR, sender_name VARCHAR, sender_email VARCHAR,
        category VARCHAR, program VARCHAR DEFAULT 'OETP', phases TEXT[],
        confidence VARCHAR, draft_text TEXT, sent_text TEXT,
        outcome VARCHAR DEFAULT 'PENDING', similarity_score FLOAT,
        top_chunks JSONB, rag_scores JSONB,
        created_at TIMESTAMP DEFAULT NOW(), resolved_at TIMESTAMP)"""))
    yield
