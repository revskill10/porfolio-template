#!/usr/bin/env python3
"""
Test code file for verifying external file loading in the code widget.
"""

import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TestResult:
    """Data class for storing test results."""
    name: str
    passed: bool
    message: str
    duration: float

class CodeWidgetTester:
    """Test class for verifying code widget functionality."""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record the result."""
        import time
        start_time = time.time()
        
        try:
            await test_func()
            duration = time.time() - start_time
            result = TestResult(test_name, True, "Test passed", duration)
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(test_name, False, str(e), duration)
        
        self.results.append(result)
        return result
    
    async def test_file_loading(self):
        """Test that files can be loaded correctly."""
        # Simulate file loading
        await asyncio.sleep(0.1)
        print("File loading test completed")
    
    async def test_syntax_highlighting(self):
        """Test syntax highlighting functionality."""
        # Simulate syntax highlighting
        await asyncio.sleep(0.05)
        print("Syntax highlighting test completed")
    
    async def test_line_numbers(self):
        """Test line number display."""
        # Simulate line number generation
        await asyncio.sleep(0.02)
        print("Line numbers test completed")
    
    async def run_all_tests(self):
        """Run all tests and return results."""
        tests = [
            ("File Loading", self.test_file_loading),
            ("Syntax Highlighting", self.test_syntax_highlighting),
            ("Line Numbers", self.test_line_numbers),
        ]
        
        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            print(f"{test_name}: {'PASS' if result.passed else 'FAIL'} ({result.duration:.3f}s)")
        
        return self.results

# Example usage
async def main():
    """Main function to run the tests."""
    tester = CodeWidgetTester()
    results = await tester.run_all_tests()
    
    # Print summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nTest Summary: {passed}/{total} tests passed")
    
    # Export results as JSON
    results_dict = {
        "summary": {"passed": passed, "total": total},
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "duration": r.duration
            }
            for r in results
        ]
    }
    
    return results_dict

if __name__ == "__main__":
    # Run the tests
    results = asyncio.run(main())
    print(json.dumps(results, indent=2))
