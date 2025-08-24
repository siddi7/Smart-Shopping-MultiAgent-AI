#!/usr/bin/env python3
"""
Test Runner for Smart Shopping Multi-Agent AI System
Provides comprehensive test execution with reporting and coverage analysis
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestRunner:
    """Comprehensive test runner for the Smart Shopping AI system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.reports_dir = project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def run_unit_tests(self, verbose=True, coverage=True):
        """Run unit tests"""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.tests_dir / "unit"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--junit-xml=" + str(self.reports_dir / "unit_tests.xml"),
            "--html=" + str(self.reports_dir / "unit_tests.html"),
            "--self-contained-html"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html:" + str(self.reports_dir / "coverage_html"),
                "--cov-report=xml:" + str(self.reports_dir / "coverage.xml"),
                "--cov-report=term-missing"
            ])
        
        return self._run_command(cmd, "Unit Tests")
    
    def run_integration_tests(self, verbose=True):
        """Run integration tests"""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.tests_dir / "integration"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--junit-xml=" + str(self.reports_dir / "integration_tests.xml"),
            "--html=" + str(self.reports_dir / "integration_tests.html"),
            "--self-contained-html"
        ]
        
        return self._run_command(cmd, "Integration Tests")
    
    def run_performance_tests(self, verbose=True):
        """Run performance tests"""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.tests_dir / "performance"),
            "-v" if verbose else "-q",
            "--tb=short",
            "-s",  # Don't capture output for performance metrics
            "--junit-xml=" + str(self.reports_dir / "performance_tests.xml"),
            "--html=" + str(self.reports_dir / "performance_tests.html"),
            "--self-contained-html"
        ]
        
        return self._run_command(cmd, "Performance Tests")
    
    def run_all_tests(self, verbose=True, coverage=True):
        """Run all tests"""
        print("🚀 Running Complete Test Suite...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.tests_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "--junit-xml=" + str(self.reports_dir / "all_tests.xml"),
            "--html=" + str(self.reports_dir / "all_tests.html"),
            "--self-contained-html"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html:" + str(self.reports_dir / "coverage_html"),
                "--cov-report=xml:" + str(self.reports_dir / "coverage.xml"),
                "--cov-report=term-missing"
            ])
        
        return self._run_command(cmd, "All Tests")
    
    def run_specific_test(self, test_path, verbose=True):
        """Run a specific test file or function"""
        print(f"🎯 Running Specific Test: {test_path}")
        
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        return self._run_command(cmd, f"Test: {test_path}")
    
    def run_tests_by_marker(self, marker, verbose=True):
        """Run tests by pytest marker"""
        print(f"🏷️  Running Tests with Marker: {marker}")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.tests_dir),
            f"-m {marker}",
            "-v" if verbose else "-q",
            "--tb=short",
            "--junit-xml=" + str(self.reports_dir / f"{marker}_tests.xml"),
            "--html=" + str(self.reports_dir / f"{marker}_tests.html"),
            "--self-contained-html"
        ]
        
        return self._run_command(cmd, f"Tests with marker: {marker}")
    
    def run_quick_test(self):
        """Run a quick smoke test"""
        print("💨 Running Quick Smoke Test...")
        
        # Run a subset of critical tests
        cmd = [
            "python", "-m", "pytest",
            str(self.tests_dir / "unit" / "test_customer_agent.py::TestCustomerAgent::test_agent_initialization"),
            str(self.tests_dir / "unit" / "test_product_agent.py::TestProductAgent::test_agent_initialization"),
            str(self.tests_dir / "unit" / "test_recommendation_agent.py::TestRecommendationAgent::test_agent_initialization"),
            str(self.tests_dir / "integration" / "test_api_integration.py::TestAPIIntegration::test_health_check"),
            "-v",
            "--tb=short"
        ]
        
        return self._run_command(cmd, "Quick Smoke Test")
    
    def check_test_dependencies(self):
        """Check if all test dependencies are installed"""
        print("📋 Checking Test Dependencies...")
        
        required_packages = [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-html",
            "pytest-xdist"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} - Missing")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n📦 Install missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ All test dependencies are installed!")
        return True
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        print("📊 Generating Test Report...")
        
        report_file = self.reports_dir / "test_summary.md"
        
        with open(report_file, "w") as f:
            f.write("# Smart Shopping Multi-Agent AI - Test Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Check for test result files
            test_files = {
                "Unit Tests": self.reports_dir / "unit_tests.xml",
                "Integration Tests": self.reports_dir / "integration_tests.xml",
                "Performance Tests": self.reports_dir / "performance_tests.xml",
                "All Tests": self.reports_dir / "all_tests.xml"
            }
            
            f.write("## Test Results\n\n")
            
            for test_type, xml_file in test_files.items():
                if xml_file.exists():
                    f.write(f"- ✅ {test_type}: [View Report]({xml_file.name})\n")
                else:
                    f.write(f"- ❌ {test_type}: Not run\n")
            
            f.write("\n## Coverage Report\n\n")
            coverage_file = self.reports_dir / "coverage.xml"
            if coverage_file.exists():
                f.write(f"- ✅ Coverage Report: [View HTML](coverage_html/index.html)\n")
            else:
                f.write("- ❌ Coverage Report: Not generated\n")
            
            f.write("\n## Test Artifacts\n\n")
            f.write("All test reports and artifacts are stored in the `test_reports/` directory:\n\n")
            
            for file_path in self.reports_dir.iterdir():
                if file_path.is_file():
                    f.write(f"- {file_path.name}\n")
        
        print(f"📋 Test report generated: {report_file}")
        return report_file
    
    def clean_reports(self):
        """Clean old test reports"""
        print("🧹 Cleaning old test reports...")
        
        if self.reports_dir.exists():
            for file_path in self.reports_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
        
        print("✅ Test reports cleaned!")
    
    def _run_command(self, cmd, test_name):
        """Run a command and return success status"""
        print(f"🔄 Executing: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ {test_name} completed successfully in {duration:.2f}s")
                return True
            else:
                print(f"❌ {test_name} failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Error running {test_name}: {e}")
            return False

def main():
    """Main entry point for the test runner"""
    parser = argparse.ArgumentParser(description="Smart Shopping AI Test Runner")
    
    parser.add_argument(
        "command",
        choices=["unit", "integration", "performance", "all", "quick", "check-deps", "clean", "report"],
        help="Test command to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--marker", "-m",
        help="Run tests with specific marker"
    )
    
    parser.add_argument(
        "--test", "-t",
        help="Run specific test file or function"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(project_root)
    
    # Execute command
    success = True
    
    if args.command == "check-deps":
        success = runner.check_test_dependencies()
    
    elif args.command == "clean":
        runner.clean_reports()
    
    elif args.command == "report":
        runner.generate_test_report()
    
    elif args.command == "quick":
        success = runner.run_quick_test()
    
    elif args.command == "unit":
        success = runner.run_unit_tests(args.verbose, not args.no_coverage)
    
    elif args.command == "integration":
        success = runner.run_integration_tests(args.verbose)
    
    elif args.command == "performance":
        success = runner.run_performance_tests(args.verbose)
    
    elif args.command == "all":
        success = runner.run_all_tests(args.verbose, not args.no_coverage)
    
    elif args.marker:
        success = runner.run_tests_by_marker(args.marker, args.verbose)
    
    elif args.test:
        success = runner.run_specific_test(args.test, args.verbose)
    
    # Generate report after running tests
    if args.command in ["unit", "integration", "performance", "all"]:
        runner.generate_test_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()