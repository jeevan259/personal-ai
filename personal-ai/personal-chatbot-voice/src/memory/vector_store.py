"""
Vector store management for memory storage
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manage vector store for memory storage"""
    
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.vector_store = None
        
    async def initialize(self):
        """Initialize vector store"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="assistant_memories",
                metadata={"description": "Personal assistant memories"}
            )
            
            logger.info(f"Vector store initialized at {self.persist_dir}")
            
        except ImportError:
            logger.warning("chromadb not installed, using mock vector store")
            self.client = None
            self.collection = None
            
    async def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None):
        """Add a memory to vector store"""
        if self.collection is None:
            await self.initialize()
            
        if self.collection is None:
            logger.warning("Cannot add memory - vector store not available")
            return False
            
        try:
            # Generate ID
            import uuid
            memory_id = str(uuid.uuid4())
            
            # Add to collection
            self.collection.add(
                documents=[content],
                metadatas=[metadata or {}],
                ids=[memory_id],
                embeddings=[embedding] if embedding else None
            )
            
            logger.debug(f"Added memory: {content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return False
            
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories"""
        if self.collection is None:
            await self.initialize()
            
        if self.collection is None:
            logger.warning("Cannot search - vector store not available")
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0])
                ):
                    memories.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": 1.0 - distance,  # Convert distance to similarity score
                        "rank": i + 1
                    })
                    
            return memories
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
            
    async def get_all_memories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all memories"""
        if self.collection is None:
            await self.initialize()
            
        if self.collection is None:
            return []
            
        try:
            # ChromaDB doesn't have a direct "get all", so we query with a neutral query
            results = self.collection.query(
                query_texts=["memory"],
                n_results=limit or 1000,
                include=["documents", "metadatas", "ids"]
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, memory_id in zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['ids'][0]
                ):
                    memories.append({
                        "id": memory_id,
                        "content": doc,
                        "metadata": metadata
                    })
                    
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
            
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        if self.collection is None:
            await self.initialize()
            
        if self.collection is None:
            return False
            
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.collection is None:
            await self.initialize()
            
        if self.collection is None:
            return {"error": "Vector store not available"}
            
        try:
            count = self.collection.count()
            return {
                "total_memories": count,
                "collection_name": self.collection.name,
                "persist_dir": str(self.persist_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {"error": str(e)}