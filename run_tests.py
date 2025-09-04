#!/usr/bin/env python3
"""Test runner script for Kangni Agents"""

import sys
import os
import subprocess

def run_quick_test():
    """Run quick test"""
    print("🚀 Running Quick Test...")
    result = subprocess.run([sys.executable, "src/tests/quick_test.py"], cwd=os.getcwd())
    return result.returncode == 0

def run_comprehensive_test():
    """Run comprehensive test"""
    print("🔍 Running Comprehensive Test...")
    result = subprocess.run([sys.executable, "src/tests/test_all.py"], cwd=os.getcwd())
    return result.returncode == 0

def run_rag_test():
    """Run RAG test"""
    print("📚 Running RAG Test...")
    result = subprocess.run([sys.executable, "src/tests/test_rag.py"], cwd=os.getcwd())
    return result.returncode == 0

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "quick"
    
    print("=" * 60)
    print("Kangni Agents Test Runner")
    print("=" * 60)
    
    success = False
    
    if test_type == "quick":
        success = run_quick_test()
    elif test_type == "all" or test_type == "comprehensive":
        success = run_comprehensive_test()
    elif test_type == "rag":
        success = run_rag_test()
    elif test_type == "all-tests":
        print("Running all test suites...")
        quick_success = run_quick_test()
        print("\n" + "="*60 + "\n")
        comprehensive_success = run_comprehensive_test()
        print("\n" + "="*60 + "\n")
        rag_success = run_rag_test()
        success = quick_success and comprehensive_success and rag_success
    else:
        print(f"Unknown test type: {test_type}")
        print("Available options: quick, all, comprehensive, rag, all-tests")
        sys.exit(1)
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
