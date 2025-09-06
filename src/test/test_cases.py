#!/usr/bin/env python3
"""
Test script for all test cases in testCases.json
测试脚本，用于测试 testCases.json 中的所有测试用例
"""

import json
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

try:
    from kangni_agents.models import UserQuery
    from kangni_agents.agents.react_agent import kangni_agent
    from kangni_agents.config import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestResult:
    def __init__(self, question: str, keywords: List[str]):
        self.question = question
        self.expected_keywords = keywords
        self.response = None
        self.success = False
        self.error = None
        self.duration = 0.0
        self.keyword_matches = []

    def check_keywords(self):
        """检查响应中是否包含预期的关键词"""
        if not self.response:
            return False
        
        response_text = ""
        if hasattr(self.response, 'answer'):
            response_text = str(self.response.answer).lower()
        elif hasattr(self.response, 'content'):
            response_text = str(self.response.content).lower()
        else:
            response_text = str(self.response).lower()
        
        for keyword in self.expected_keywords:
            if keyword.lower() in response_text:
                self.keyword_matches.append(keyword)
        
        return len(self.keyword_matches) > 0

    def to_dict(self):
        return {
            "question": self.question,
            "expected_keywords": self.expected_keywords,
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "duration": self.duration,
            "keyword_matches": self.keyword_matches,
            "response": str(self.response) if self.response else None
        }

async def run_single_test(question: str, keywords: List[str]) -> TestResult:
    """运行单个测试用例"""
    result = TestResult(question, keywords)
    
    try:
        logger.info(f"Testing: {question[:50]}...")
        start_time = time.time()
        
        # Create query
        query = UserQuery(question=question)
        
        # Execute query
        response = await kangni_agent.query(
            question=query.question,
            context=query.context
        )
        
        result.duration = time.time() - start_time
        result.response = response
        
        # Check if keywords are present in response
        result.success = result.check_keywords()
        
        logger.info(f"✅ Test completed in {result.duration:.2f}s - {'PASS' if result.success else 'FAIL'}")
        
    except Exception as e:
        result.duration = time.time() - start_time
        result.error = e
        logger.error(f"❌ Test failed: {e}")
    
    return result

async def run_all_tests():
    """运行所有测试用例"""
    # Load test cases
    test_cases_path = Path("src/test/data/test_cases.json")
    if not test_cases_path.exists():
        print(f"❌ Test cases file not found: {test_cases_path}")
        return
    
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    print(f"🚀 Running {len(test_cases)} test cases...")
    print(f"📊 Configuration:")
    print(f"   RAGFlow Server: {settings.ragflow_mcp_server_url}")
    print(f"   Default Dataset: {settings.ragflow_default_dataset_id}")
    print("=" * 80)
    
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case.get("question", "")
        keywords = test_case.get("keywords", [])
        
        print(f"\n[{i}/{len(test_cases)}] {question}")
        print(f"Expected keywords: {', '.join(keywords)}")
        
        result = await run_single_test(question, keywords)
        results.append(result)
        
        if result.success:
            passed += 1
            print(f"✅ PASS - Found keywords: {', '.join(result.keyword_matches)}")
        else:
            failed += 1
            if result.error:
                print(f"❌ ERROR - {result.error}")
            else:
                print(f"❌ FAIL - No expected keywords found in response")
        
        print(f"Duration: {result.duration:.2f}s")
        
        # Small delay between tests to avoid overwhelming the system
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📈 TEST SUMMARY")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(test_cases)*100:.1f}%)")
    print("=" * 80)
    
    # Save detailed results
    results_file = Path("test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    # Print failed cases for review
    if failed > 0:
        print(f"\n❌ FAILED CASES:")
        for result in results:
            if not result.success:
                print(f"  • {result.question}")
                if result.error:
                    print(f"    Error: {result.error}")
                else:
                    print(f"    Expected: {', '.join(result.expected_keywords)}")
                    print(f"    Found: {', '.join(result.keyword_matches) if result.keyword_matches else 'None'}")

def main():
    """主函数"""
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()