#!/usr/bin/env python3
"""
Comprehensive Vector Embedding Test Suite
Combines basic embedding generation and advanced vector operations testing.
"""
import asyncio
import json
import os
import sys
import httpx
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载.env文件: {env_path}")
else:
    print(f"⚠️  未找到.env文件: {env_path}")

from src.kangni_agents.services.vector_embedding_service import vector_service
from src.kangni_agents.services.database_service import db_service
from src.kangni_agents.utils.query_preprocessor import QueryPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveVectorEmbeddingTest:
    """Comprehensive test suite for vector embedding functionality"""
    
    def __init__(self):
        self.query_preprocessor = QueryPreprocessor(vector_service)
        self.test_results = []
        
        # Get embedding configuration from environment
        self.embedding_base_url = os.getenv('EMBEDDING_BASE_URL', 'http://158.158.4.66:4432/v1')
        self.embedding_api_key = os.getenv('EMBEDDING_API_KEY')
        
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY not found in environment variables")
    
    async def test_basic_embedding_generation(self):
        """Test basic embedding generation using BGE-M3 model"""
        print("\n=== Testing Basic Embedding Generation ===")
        try:
            # Test text for embedding generation
            test_text = "This is a test sentence for embedding generation."
            
            print(f"Generating embedding for: '{test_text}'")
            print(f"Using API: {self.embedding_base_url}")
            
            # Generate embedding using the service
            embedding = await vector_service.generate_embedding(test_text)
            
            # Validate embedding
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) == 1024, f"Embedding dimension should be 1024, got {len(embedding)}"
            assert all(isinstance(x, float) for x in embedding), "All embedding values should be float"
            
            print(f"✓ Successfully generated embedding with {len(embedding)} dimensions")
            print(f"✓ First 5 values: {embedding[:5]}")
            
            self.test_results.append(("test_basic_embedding_generation", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ Failed to generate embedding: {e}")
            self.test_results.append(("test_basic_embedding_generation", f"FAILED: {e}"))
            return False
    
    async def test_direct_api_embedding(self):
        """Test direct API call to BGE-M3 embedding service"""
        print("\n=== Testing Direct API Embedding ===")
        try:
            test_text = "南京塞尔塔门系统项目"
            
            # Prepare request to BGE-M3 embedding API
            headers = {
                'Authorization': f'Bearer {self.embedding_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'bge-m3',
                'input': test_text,
                'encoding_format': 'float'
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f'{self.embedding_base_url}/embeddings',
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        embedding = data['data'][0].get('embedding', [])
                        
                        if embedding:
                            print(f"✓ Direct API call successful")
                            print(f"✓ Generated embedding with dimension: {len(embedding)}")
                            print(f"✓ Sample values: {embedding[:5]}")
                            
                            self.test_results.append(("test_direct_api_embedding", "PASSED"))
                            return True
                        else:
                            raise ValueError("No embedding returned from BGE-M3 model")
                    else:
                        raise ValueError("Invalid response format from BGE-M3 model")
                else:
                    raise Exception(f"API error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            print(f"✗ Direct API call failed: {e}")
            self.test_results.append(("test_direct_api_embedding", f"FAILED: {e}"))
            return False
    
    async def test_store_field_embedding(self):
        """Test storing field embeddings"""
        print("\n=== Testing Store Field Embedding ===")
        try:
            table_name = "kn_quality_trace_prod_order"
            field_name = "projectname_s"
            field_value = "测试项目_" + str(int(asyncio.get_event_loop().time()))
            
            # Store embedding
            embedding_id = await vector_service.store_field_embedding(
                table_name, field_name, field_value
            )
            
            assert isinstance(embedding_id, int), "Embedding ID should be an integer"
            assert embedding_id > 0, "Embedding ID should be positive"
            
            print(f"✓ Stored embedding for '{field_value}'")
            print(f"  Table: {table_name}")
            print(f"  Field: {field_name}")
            print(f"  Embedding ID: {embedding_id}")
            
            # Test duplicate handling
            embedding_id2 = await vector_service.store_field_embedding(
                table_name, field_name, field_value
            )
            
            assert embedding_id == embedding_id2, "Duplicate values should return same ID"
            print(f"✓ Duplicate handling works correctly")
            
            self.test_results.append(("test_store_field_embedding", "PASSED"))
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Unknown PG numeric type" in error_msg:
                print(f"✗ PostgreSQL vector type error: {error_msg}")
                print("  This suggests pgvector extension may not be properly installed")
                print("  or the vector column type is not recognized")
                self.test_results.append(("test_store_field_embedding", f"FAILED: PostgreSQL vector type error - {error_msg}"))
            else:
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
                await vector_service.store_field_embedding(
                    table_name, field_name, value
                )
            
            print(f"✓ Stored {len(test_values)} test values")
            
            # Search for similar values
            search_text = "门系统"
            results = await vector_service.search_similar_values(
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
            error_msg = str(e)
            if "Unknown PG numeric type" in error_msg:
                print(f"✗ PostgreSQL vector type error: {error_msg}")
                print("  This suggests pgvector extension may not be properly installed")
                print("  or the vector column type is not recognized")
                self.test_results.append(("test_search_similar_values", f"FAILED: PostgreSQL vector type error - {error_msg}"))
            else:
                print(f"✗ Failed similarity search: {e}")
                self.test_results.append(("test_search_similar_values", f"FAILED: {e}"))
            return False
    
    async def test_sync_table_data(self):
        """Test syncing table data to embeddings"""
        print("\n=== Testing Sync Table Data ===")
        try:
            # This test requires actual data in the business table
            # We'll test with a small limit to avoid long processing
            result = await vector_service.sync_table_field_data(
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
            base_url = "http://localhost:8000/qomo/v1/embeddings"
            test_passed = True
            
            async with httpx.AsyncClient() as client:
                # Test sync endpoint
                print("Testing /values endpoint...")
                response = await client.post(
                    f"{base_url}/values",
                    params={"limit": 2}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Sync endpoint working")
                    print(f"  Response: {data}")
                else:
                    print(f"✗ Sync endpoint failed: {response.status_code} - {response.text}")
                    test_passed = False
                
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
                    print(f"✗ Search endpoint failed: {response.status_code} - {response.text}")
                    test_passed = False
                
                if test_passed:
                    self.test_results.append(("test_api_endpoints", "PASSED"))
                    return True
                else:
                    self.test_results.append(("test_api_endpoints", "FAILED: One or more endpoints returned error status"))
                    return False
                
        except httpx.ConnectError:
            print("ℹ API server not running, skipping API tests")
            self.test_results.append(("test_api_endpoints", "SKIPPED: Server not running"))
            return True
        except Exception as e:
            print(f"✗ Failed API endpoint tests: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            self.test_results.append(("test_api_endpoints", f"FAILED: {e}"))
            return False
    
    async def test_embedding_configuration(self):
        """Test embedding configuration and environment setup"""
        print("\n=== Testing Embedding Configuration ===")
        try:
            print(f"✓ Embedding Base URL: {self.embedding_base_url}")
            print(f"✓ API Key configured: {'Yes' if self.embedding_api_key else 'No'}")
            print(f"✓ API Key length: {len(self.embedding_api_key) if self.embedding_api_key else 0}")
            
            # Test API connectivity
            headers = {
                'Authorization': f'Bearer {self.embedding_api_key}',
                'Content-Type': 'application/json'
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test with a simple request
                response = await client.get(
                    f'{self.embedding_base_url}/models',
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"✓ API connectivity test passed")
                    data = response.json()
                    if 'data' in data:
                        print(f"✓ Available models: {len(data['data'])}")
                else:
                    print(f"⚠ API connectivity test returned: {response.status_code}")
            
            self.test_results.append(("test_embedding_configuration", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ Configuration test failed: {e}")
            self.test_results.append(("test_embedding_configuration", f"FAILED: {e}"))
            return False
    
    async def test_postgresql_vector_extension(self):
        """Test PostgreSQL vector extension status"""
        print("\n=== Testing PostgreSQL Vector Extension ===")
        try:
            from sqlalchemy import text
            
            with vector_service.SessionLocal() as session:
                # Check if vector extension is installed
                result = session.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).fetchone()
                
                if result:
                    print("✓ pgvector extension is installed")
                    
                    # Check if vector type is available
                    result = session.execute(text("SELECT 1 FROM pg_type WHERE typname = 'vector'")).fetchone()
                    if result:
                        print("✓ vector type is available")
                    else:
                        print("✗ vector type is not available")
                        self.test_results.append(("test_postgresql_vector_extension", "FAILED: vector type not available"))
                        return False
                        
                    # Test vector operations
                    try:
                        result = session.execute(text("SELECT '[1,2,3]'::vector <-> '[1,2,3]'::vector as distance")).fetchone()
                        if result:
                            print(f"✓ Vector operations working (test distance: {result[0]})")
                        else:
                            print("✗ Vector operations not working")
                            self.test_results.append(("test_postgresql_vector_extension", "FAILED: vector operations not working"))
                            return False
                    except Exception as e:
                        print(f"✗ Vector operations failed: {e}")
                        self.test_results.append(("test_postgresql_vector_extension", f"FAILED: vector operations error - {e}"))
                        return False
                        
                else:
                    print("✗ pgvector extension is not installed")
                    print("  Please install pgvector extension: CREATE EXTENSION vector;")
                    self.test_results.append(("test_postgresql_vector_extension", "FAILED: pgvector extension not installed"))
                    return False
            
            self.test_results.append(("test_postgresql_vector_extension", "PASSED"))
            return True
            
        except Exception as e:
            print(f"✗ PostgreSQL vector extension test failed: {e}")
            self.test_results.append(("test_postgresql_vector_extension", f"FAILED: {e}"))
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 80)
        print("Starting Comprehensive Vector Embedding Tests")
        print("=" * 80)
        
        # Run tests in logical order
        await self.test_embedding_configuration()
        await self.test_postgresql_vector_extension()
        await self.test_basic_embedding_generation()
        await self.test_direct_api_embedding()
        await self.test_store_field_embedding()
        await self.test_search_similar_values()
        await self.test_sync_table_data()
        await self.test_query_preprocessor_with_vector()
        await self.test_api_endpoints()
        
        # Print summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        
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
    try:
        # Set environment for testing
        os.environ['DB_TYPE'] = 'postgresql'
        
        tester = ComprehensiveVectorEmbeddingTest()
        success = await tester.run_all_tests()
        
        if success:
            print("\n✅ All tests completed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
