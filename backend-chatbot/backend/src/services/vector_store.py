from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from config.config import qdrant_client
import logging
import traceback
from qdrant_client.http.models import NearestQuery

# Initialize logger
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Wrapper for Qdrant operations (create_collection if not exists, upsert, search top-k)
    """
    
    def __init__(self, collection_name: str = "book_collection"):
        self.collection_name = collection_name
        self.client = qdrant_client
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        Ensure the collection exists with the proper configuration
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
        else:
            print(f"Collection '{self.collection_name}' already exists.")
         
    def upsert_vectors(self, points: List[Dict[str, Any]]):
        """
        Upsert vectors to the Qdrant collection
        Each point should have: id, vector, payload
        """
        try:
            # Prepare points in the required format
            points_list = [
                models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {})
                )
                for point in points
            ]
            
            # Upsert the points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_list,
                wait=True  # Wait for operation to complete
            )
            
            logger.info(f"Successfully upserted {len(points)} vectors to collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False
    
    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection using the new query_points API
        """
        try:
            from qdrant_client.http.models import NearestQuery
            
            # New way: use query_points with nearest query
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=NearestQuery(nearest=query_vector),
                limit=top_k
            ).points  # .points directly gives the list of ScoredPoint
            
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload or {},
                    "text": hit.payload.get("text", "") if hit.payload else ""
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar vectors for query")
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    def delete_collection(self):
        """
        Delete the entire collection (use with caution!)
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific point by ID
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {str(e)}")
            return None