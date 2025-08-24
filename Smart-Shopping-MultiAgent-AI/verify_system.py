#!/usr/bin/env python3
"""
Smart Shopping Multi-Agent AI System - Verification & Demo Script
Quick verification that the system is working properly
"""

import sys
import os
import asyncio
from pathlib import Path

def check_project_structure():
    """Verify project structure is correct"""
    print("🔍 Checking project structure...")
    
    required_dirs = [
        "src/agents",
        "src/database", 
        "src/utils",
        "config",
        "tests",
        "dashboard",
        "scripts"
    ]
    
    required_files = [
        "main.py",
        "demo.py", 
        "requirements.txt",
        "README.md",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Missing")
            return False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing")
            return False
    
    return True

def test_imports():
    """Test that core modules can be imported"""
    print("\n🧪 Testing core imports...")
    
    try:
        sys.path.append(str(Path.cwd()))
        
        # Test basic imports
        from src.utils.config import Settings
        print("  ✅ Config module imports successfully")
        
        from src.database.models import DatabaseManager
        print("  ✅ Database models import successfully")
        
        from src.agents.base_agent import BaseAgent
        print("  ✅ Base agent imports successfully")
        
        # Test settings
        settings = Settings()
        print(f"  ✅ Settings initialized: {settings.APP_NAME}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_fastapi_app():
    """Test that FastAPI app can be created"""
    print("\n🌐 Testing FastAPI application...")
    
    try:
        from main import app
        print("  ✅ FastAPI app creates successfully")
        
        # Test some basic app properties
        if hasattr(app, 'title'):
            print(f"  ✅ App title: {app.title}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FastAPI error: {e}")
        return False

def test_demo_system():
    """Test that demo system initializes"""
    print("\n🎮 Testing demo system...")
    
    try:
        from demo import SmartShoppingDemo
        demo = SmartShoppingDemo()
        print("  ✅ Demo system initializes successfully")
        
        # Test basic demo functionality
        if hasattr(demo, 'agents'):
            print("  ✅ Demo has agents initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Demo error: {e}")
        return False

async def test_agent_creation():
    """Test agent creation and basic functionality"""
    print("\n🤖 Testing agent creation...")
    
    try:
        from src.agents.customer_agent import CustomerAgent
        from src.agents.product_agent import ProductAgent  
        from src.agents.recommendation_agent import RecommendationAgent
        from src.database.models import DatabaseManager
        from src.utils.config import Settings
        
        settings = Settings()
        db_manager = DatabaseManager("sqlite:///test_verification.db")
        
        # Test agent creation
        customer_agent = CustomerAgent("test_customer", db_manager, {"settings": settings})
        print("  ✅ Customer agent created successfully")
        
        product_agent = ProductAgent("test_product", db_manager, {"settings": settings})
        print("  ✅ Product agent created successfully")
        
        rec_agent = RecommendationAgent("test_rec", db_manager, {"settings": settings})
        print("  ✅ Recommendation agent created successfully")
        
        # Test basic agent functionality
        metrics = customer_agent.get_performance_metrics()
        if metrics:
            print("  ✅ Agent performance metrics working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent creation error: {e}")
        return False

def test_github_deployment():
    """Verify GitHub deployment files"""
    print("\n🚀 Checking GitHub deployment...")
    
    github_files = [
        ".github/workflows/test-deployment.yml",
        ".gitignore",
        ".env.example"
    ]
    
    for file_path in github_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing")
            return False
    
    return True

def display_usage_instructions():
    """Display usage instructions"""
    print("\n" + "="*60)
    print("🛍️  SMART SHOPPING MULTI-AGENT AI SYSTEM")
    print("="*60)
    print("\n📚 QUICK START GUIDE:")
    print("\n1. 🎮 Run Interactive Demo:")
    print("   python demo.py")
    
    print("\n2. 🌐 Start API Server:")
    print("   python main.py")
    print("   # OR using uvicorn:")
    print("   python -m uvicorn main:app --reload --port 8000")
    
    print("\n3. 📊 Access Dashboard:")
    print("   http://localhost:8000/dashboard")
    
    print("\n4. 📖 View API Documentation:")
    print("   http://localhost:8000/docs")
    
    print("\n5. 🧪 Run Tests:")
    print("   python tests/run_tests.py all")
    
    print("\n6. 🐳 Docker Deployment:")
    print("   docker-compose up")
    
    print("\n🔗 GitHub Repository:")
    print("   https://github.com/siddi7/Smart-Shopping-MultiAgent-AI")
    
    print("\n" + "="*60)

def main():
    """Main verification function"""
    print("🔥 Smart Shopping Multi-Agent AI System - Verification")
    print("="*55)
    
    tests = [
        ("Project Structure", check_project_structure),
        ("Core Imports", test_imports),
        ("FastAPI Application", test_fastapi_app),
        ("Demo System", test_demo_system), 
        ("GitHub Deployment", test_github_deployment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Test async agent creation
    try:
        print("\n🤖 Testing agent creation...")
        result = asyncio.run(test_agent_creation())
        results.append(("Agent Creation", result))
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")
        results.append(("Agent Creation", False))
    
    # Display results
    print("\n" + "="*55)
    print("📊 VERIFICATION RESULTS")
    print("="*55)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment!")
        display_usage_instructions()
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)