"""Optional vector memory layer for historical trading context retrieval."""

from services.vector_memory.service import VectorMemoryService, get_vector_memory_service

__all__ = ["VectorMemoryService", "get_vector_memory_service"]
