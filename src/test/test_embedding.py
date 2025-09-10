import httpx
import os
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_embedding(text):
    """
    Generate embedding using BGE-M3 model
    """
    # BGE-M3 embedding model configuration
    embedding_base_url = os.getenv('EMBEDDING_BASE_URL', 'http://158.158.4.66:4432/v1')
    embedding_api_key = os.getenv('EMBEDDING_API_KEY', 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5YjljMmRkYmVkODc0NWFlYWI0NWE3OTZmZGZhYTc4NSIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc1MDczOTk5NSIsImV4cCI6IjE3NTA3NDM1OTUifQ.hN_vKo4rAbEm9qk2z8r4YgEqSEyaNM4UdY0ytbywvz9QkgmAD-t5sd3OYlQYzc2OZcO4YvGBJIJlunMIruNVSUXT3sdytoifHF_h3PtlTl-L8noogKKQDAfYyuyElE6zx2BA7u6EMhc2GWI0Nj8a3Psq_9DRb3vqJZKZDi732JmUXjNfdvYxzkNtiyO4Adb-06dgt8jjvCr0z2pwoIZG8ke5qZQcRp7huBI77Y1_uzVhpCOtXnjOqtZFQtd2U9jfhLk6dPrdhljWrnePUQ7WAvFji9JPKSZJLUVErX3ADYMt6FCwgy1XWYNGYk0xa0W4kBdX9H1Cgzg8cHlM00_NUg')
    
    # Prepare request to BGE-M3 embedding API
    headers = {
        'Authorization': f'Bearer {embedding_api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'bge-m3',
        'input': text,
        'encoding_format': 'float'
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f'{embedding_base_url}/embeddings',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            # Extract embedding from response
            # BGE-M3 returns embeddings in OpenAI-compatible format
            if 'data' in data and len(data['data']) > 0:
                embedding = data['data'][0].get('embedding', [])
                
                if embedding:
                    logger.info(f"Generated embedding with dimension: {len(embedding)}")
                    return embedding
                else:
                    raise ValueError("No embedding returned from BGE-M3 model")
            else:
                raise ValueError("Invalid response format from BGE-M3 model")
        else:
            logger.error(f"BGE-M3 API error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to generate embedding: {response.status_code}")

async def main():
    """
    Main function to test the embedding generation
    """
    # Test text for embedding generation
    test_text = "This is a test sentence for embedding generation."
    
    try:
        print(f"Generating embedding for: '{test_text}'")
        embedding = await generate_embedding(test_text)
        print(f"Successfully generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error generating embedding: {e}")

if __name__ == "__main__":
    asyncio.run(main())