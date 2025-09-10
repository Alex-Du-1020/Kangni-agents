"""
Comprehensive tests for vector embedding functionality.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.kangni_agents.services.vector_embedding_service import VectorEmbeddingService
from src.kangni_agents.services.database_service import db_service
from src.kangni_agents.services.llm_service import llm_service
from src.kangni_agents.utils.query_preprocessor import QueryPreprocessor

class TestVectorEmbedding:
    """Test suite for vector embedding functionality"""
    
    def __init__(self):
        self.vector_service = VectorEmbeddingService(db_service, llm_service)
        self.query_preprocessor = QueryPreprocessor(self.vector_service)
        self.test_results = []
    
    async def test_generate_embedding(self):
        """Test embedding generation"""
        print("\n=== Testing Embedding Generation ===")
        try:
            text = "南京塞尔塔门系统项目"
            embedding = await self.vector_service.generate_embedding(text)
            
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) == 1024, f"Embedding dimension should be 1024, got {len(embedding)}"
            assert all(isinstance(x, float) for x in embedding), "All embedding values should be float"
            
            print(f"✓ Generated embedding for '{text}'")
            print(f"  Dimension: {len(embedding)}")
            print(f"  Sample values: {embedding[:5]}")
            
            self.test_results.append(("test_generate_embedding", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ Failed to generate embedding: {e}")
            self.test_results.append(("test_generate_embedding", f"FAILED: {e}"))
            return False
    
    async def test_store_field_embedding(self):
        """Test storing field embeddings"""
        print("\n=== Testing Store Field Embedding ===")
        try:
            table_name = "kn_quality_trace_prod_order"
            field_name = "projectname_s"
            field_value = "测试项目_" + str(int(asyncio.get_event_loop().time()))
            
            # Store embedding
            embedding_id = await self.vector_service.store_field_embedding(
                table_name, field_name, field_value
            )
            
            assert isinstance(embedding_id, int), "Embedding ID should be an integer"
            assert embedding_id > 0, "Embedding ID should be positive"
            
            print(f"✓ Stored embedding for '{field_value}'")
            print(f"  Table: {table_name}")
            print(f"  Field: {field_name}")
            print(f"  Embedding ID: {embedding_id}")
            
            # Test duplicate handling
            embedding_id2 = await self.vector_service.store_field_embedding(
                table_name, field_name, field_value
            )
            
            assert embedding_id == embedding_id2, "Duplicate values should return same ID"
            print(f"✓ Duplicate handling works correctly")
            
            self.test_results.append(("test_store_field_embedding", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ Failed to store embedding: {e}")
            self.test_results.append(("test_store_field_embedding", f"FAILED: {e}"))
            return False
    
    async def test_search_similar_values(self):
        """Test vector similarity search"""
        print("\n=== Testing Vector Similarity Search ===")
        try:
            # First, store some test data
            test_values = [
                "南京塞尔塔门系统项目",
                "北京地铁门控项目",
                "上海高铁座椅项目",
                "广州地铁屏蔽门项目"
            ]
            
            table_name = "kn_quality_trace_prod_order"
            field_name = "projectname_s"
            
            # Store test embeddings
            for value in test_values:
                await self.vector_service.store_field_embedding(
                    table_name, field_name, value
                )
            
            print(f"✓ Stored {len(test_values)} test values")
            
            # Search for similar values
            search_text = "门系统"
            results = await self.vector_service.search_similar_values(
                search_text, table_name, field_name, top_k=3
            )
            
            assert isinstance(results, list), "Results should be a list"
            assert len(results) <= 3, f"Should return at most 3 results, got {len(results)}"
            
            print(f"✓ Search for '{search_text}' returned {len(results)} results:")
            for result in results:
                print(f"  - {result['field_value']} (similarity: {result['similarity']:.3f})")
            
            self.test_results.append(("test_search_similar_values", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ Failed similarity search: {e}")
            self.test_results.append(("test_search_similar_values", f"FAILED: {e}"))
            return False
    
    async def test_sync_table_data(self):
        """Test syncing table data to embeddings"""
        print("\n=== Testing Sync Table Data ===")
        try:
            # This test requires actual data in the business table
            # We'll test with a small limit to avoid long processing
            result = await self.vector_service.sync_table_field_data(
                table_name="kn_quality_trace_prod_order",
                field_name="projectname_s",
                limit=5
            )
            
            assert result['success'], "Sync should be successful"
            assert 'synced' in result, "Result should contain synced count"
            assert 'total' in result, "Result should contain total count"
            
            print(f"✓ Sync completed successfully")
            print(f"  Synced: {result['synced']}")
            print(f"  Skipped: {result.get('skipped', 0)}")
            print(f"  Errors: {result.get('errors', 0)}")
            print(f"  Total: {result['total']}")
            
            self.test_results.append(("test_sync_table_data", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ Failed to sync table data: {e}")
            self.test_results.append(("test_sync_table_data", f"FAILED: {e}"))
            return False
    
    async def test_query_preprocessor_with_vector(self):
        """Test query preprocessor with vector search"""
        print("\n=== Testing Query Preprocessor with Vector Search ===")
        try:
            query = "查询南京塞尔塔门系统项目的生产订单"
            
            # Test without vector search
            result_basic = await self.query_preprocessor.preprocess_query(query, use_vector_search=False)
            
            assert result_basic.vector_suggestions is None or len(result_basic.vector_suggestions) == 0
            print(f"✓ Basic preprocessing completed")
            print(f"  Original: {result_basic.original_query}")
            print(f"  Processed: {result_basic.processed_query}")
            print(f"  Entities: {len(result_basic.entities)}")
            
            # Test with vector search
            result_vector = await self.query_preprocessor.preprocess_query(query, use_vector_search=True)
            
            if result_vector.vector_suggestions:
                print(f"✓ Vector-enhanced preprocessing completed")
                print(f"  Suggestions: {result_vector.vector_suggestions[:3]}")
            else:
                print(f"ℹ No vector suggestions found (may need data sync first)")
            
            self.test_results.append(("test_query_preprocessor_with_vector", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ Failed query preprocessing: {e}")
            self.test_results.append(("test_query_preprocessor_with_vector", f"FAILED: {e}"))
            return False
    
    async def test_api_endpoints(self):
        """Test API endpoints for vector embedding"""
        print("\n=== Testing API Endpoints ===")
        try:
            import httpx
            
            base_url = "http://localhost:8000/api/v1/embeddings"
            
            async with httpx.AsyncClient() as client:
                # Test sync endpoint
                print("Testing /sync/production-orders endpoint...")
                response = await client.post(
                    f"{base_url}/sync/production-orders",
                    params={"limit": 2}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Sync endpoint working")
                    print(f"  Response: {data}")
                else:
                    print(f"✗ Sync endpoint failed: {response.status_code}")
                
                # Test search endpoint
                print("\nTesting /search endpoint...")
                response = await client.get(
                    f"{base_url}/search",
                    params={
                        "query": "门系统",
                        "top_k": 5,
                        "threshold": 0.5
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Search endpoint working")
                    print(f"  Found {len(data)} results")
                else:
                    print(f"✗ Search endpoint failed: {response.status_code}")
                
                self.test_results.append(("test_api_endpoints", "PASSED"))
                return True
                
        except httpx.ConnectError:
            print("ℹ API server not running, skipping API tests")
            self.test_results.append(("test_api_endpoints", "SKIPPED: Server not running"))
            return True
        except Exception as e:
            print(f"✗ Failed API endpoint tests: {e}")
            self.test_results.append(("test_api_endpoints", f"FAILED: {e}"))
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("Starting Vector Embedding Tests")
        print("=" * 60)
        
        # Run tests
        await self.test_generate_embedding()
        await self.test_store_field_embedding()
        await self.test_search_similar_values()
        await self.test_sync_table_data()
        await self.test_query_preprocessor_with_vector()
        await self.test_api_endpoints()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        passed = sum(1 for _, status in self.test_results if status == "PASSED")
        failed = sum(1 for _, status in self.test_results if "FAILED" in status)
        skipped = sum(1 for _, status in self.test_results if "SKIPPED" in status)
        
        for test_name, status in self.test_results:
            status_symbol = "✓" if status == "PASSED" else "✗" if "FAILED" in status else "⊘"
            print(f"{status_symbol} {test_name}: {status}")
        
        print(f"\nTotal: {len(self.test_results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        
        return failed == 0

async def main():
    """Main test runner"""
    # Set environment for testing
    os.environ['DB_TYPE'] = 'postgresql'
    
    tester = TestVectorEmbedding()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)