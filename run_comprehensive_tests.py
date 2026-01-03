#!/usr/bin/env python3
"""
Quick test runner for the comprehensive test suite
Provides easy commands to run different test categories
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return exit code"""
    print(f"\n{'='*70}")
    print(f"🧪 {description}")
    print(f"{'='*70}")

    # Replace 'python' with the current Python interpreter to respect venv
    if cmd[0] == 'python':
        cmd[0] = sys.executable

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    """Main test runner"""
    # Show which Python interpreter is being used
    print(f"🐍 Using Python: {sys.executable}")
    print(f"📍 Python version: {sys.version.split()[0]}\n")

    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║              Comprehensive Test Suite Runner                         ║
╚══════════════════════════════════════════════════════════════════════╝

Usage: python run_comprehensive_tests.py [COMMAND]

Available Commands:

  📦 DATABASE TESTS
    database        - Test database creation and integrity (6 tests)

  🧠 RETRIEVER TESTS
    smart          - Test smart retriever (LLM disambiguation) (5 tests)
    contextual     - Test contextual retriever (query reformulation) (7 tests)
    combined       - Test combined retrievers (2 tests)
    retrievers     - Run all retriever tests (14 tests)

  🔧 PIPELINE TESTS
    pipeline       - Test pipeline configuration (5 tests)
    integration    - Run all integration tests (~13 tests)

  🚀 END-TO-END TESTS
    e2e            - Run E2E tests (requires API key + DB) (6 tests)

  📊 COMPREHENSIVE
    all            - Run ALL tests in comprehensive suite (31 tests)
    unit           - Run all unit tests (12 tests)
    fast           - Run all tests except E2E (25 tests)

  📈 COVERAGE
    coverage       - Run ALL tests with text coverage report
    coverage-html  - Run ALL tests and generate HTML coverage (open htmlcov/index.html)
    coverage-all   - Run ALL test files with HTML coverage

  🔍 SPECIFIC FILES
    test-pipeline       - Run updated test_pipeline.py
    test-dual          - Run updated test_dual_retriever.py

Examples:
    python run_comprehensive_tests.py database
    python run_comprehensive_tests.py smart
    python run_comprehensive_tests.py all
    python run_comprehensive_tests.py coverage

For more information, see: tests/QUICK_START.md
        """)
        return 1

    command = sys.argv[1].lower()

    # Base test file
    test_file = "tests/test_pipeline_comprehensive.py"

    # Command mapping
    commands = {
        # Database tests
        "database": (
            ["python", "-m", "pytest", f"{test_file}::TestDatabaseCreation", "-v"],
            "Testing database creation and integrity"
        ),

        # Retriever tests
        "smart": (
            ["python", "-m", "pytest", f"{test_file}::TestSmartRetriever", "-v"],
            "Testing smart retriever (LLM disambiguation)"
        ),
        "contextual": (
            ["python", "-m", "pytest", f"{test_file}::TestContextualRetriever", "-v"],
            "Testing contextual retriever (query reformulation)"
        ),
        "combined": (
            ["python", "-m", "pytest", f"{test_file}::TestCombinedRetrievers", "-v"],
            "Testing combined retriever layers"
        ),
        "retrievers": (
            ["python", "-m", "pytest", f"{test_file}::TestSmartRetriever",
             f"{test_file}::TestContextualRetriever",
             f"{test_file}::TestCombinedRetrievers", "-v"],
            "Testing all retriever components"
        ),

        # Pipeline tests
        "pipeline": (
            ["python", "-m", "pytest", f"{test_file}::TestRAGPipelineIntegration", "-v"],
            "Testing pipeline configuration and setup"
        ),
        "integration": (
            ["python", "-m", "pytest", "-m", "integration", "-v"],
            "Running all integration tests"
        ),

        # E2E tests
        "e2e": (
            ["python", "-m", "pytest", f"{test_file}::TestEndToEnd", "-v"],
            "Running end-to-end tests (requires API key and database)"
        ),

        # Comprehensive
        "all": (
            ["python", "-m", "pytest", test_file, "-v"],
            "Running ALL comprehensive tests"
        ),
        "unit": (
            ["python", "-m", "pytest", "-m", "unit", "-v"],
            "Running all unit tests"
        ),
        "fast": (
            ["python", "-m", "pytest", "-m", "not e2e and not slow", "-v"],
            "Running all fast tests (excluding E2E)"
        ),

        # Coverage
        "coverage": (
            ["python", "-m", "pytest", test_file, "--cov=src", "--cov-report=term-missing", "-v"],
            "Running ALL tests with coverage report"
        ),
        "coverage-html": (
            ["python", "-m", "pytest", test_file, "--cov=src", "--cov-report=html", "-v"],
            "Running ALL tests and generating HTML coverage report"
        ),
        "coverage-all": (
            ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=html", "-v"],
            "Running ALL test files with HTML coverage report"
        ),

        # Specific files
        "test-pipeline": (
            ["python", "-m", "pytest", "tests/test_pipeline.py", "-v"],
            "Running updated test_pipeline.py"
        ),
        "test-dual": (
            ["python", "-m", "pytest", "tests/test_dual_retriever.py", "-v"],
            "Running updated test_dual_retriever.py"
        ),
    }

    if command not in commands:
        print(f"❌ Unknown command: {command}")
        print(f"\nRun without arguments to see available commands.")
        return 1

    cmd, description = commands[command]
    exit_code = run_command(cmd, description)

    if exit_code == 0:
        print(f"\n✅ SUCCESS! All tests passed.")

        # Special message for coverage reports
        if command == "coverage-html":
            print(f"\n📊 Coverage report generated at: htmlcov/index.html")
            print(f"   Open with your browser to view detailed coverage.")
    else:
        print(f"\n❌ FAILED! Some tests did not pass.")
        print(f"   Review the output above for details.")

    print(f"\n{'='*70}\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
