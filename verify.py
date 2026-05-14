#!/usr/bin/env python
"""
Verification Script for Evolutionary ML Engine

Runs a quick test of all major components to ensure everything is working
before hackathon submission. Exits with status 0 if all tests pass.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print()
    print("=" * 80)
    print(text.center(80))
    print("=" * 80)
    print()


def print_status(test_name, passed):
    """Print test status."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{test_name:<60} {status}")
    return passed


def test_imports():
    """Test that all required imports work."""
    print_header("Testing Required Imports")
    
    all_passed = True
    
    try:
        import numpy
        all_passed &= print_status("NumPy", True)
    except ImportError:
        all_passed &= print_status("NumPy", False)
    
    try:
        import pandas
        all_passed &= print_status("Pandas", True)
    except ImportError:
        all_passed &= print_status("Pandas", False)
    
    try:
        import scipy
        all_passed &= print_status("SciPy", True)
    except ImportError:
        all_passed &= print_status("SciPy", False)
    
    try:
        import sklearn
        all_passed &= print_status("scikit-learn", True)
    except ImportError:
        all_passed &= print_status("scikit-learn", False)
    
    return all_passed


def test_core_engine():
    """Test the core evolutionary ML engine."""
    print_header("Testing Core Engine Components")
    
    all_passed = True
    
    try:
        from evolutionary_ml_engine import (
            HPGenome, Worker, PopulationAutopilot, ConformalCalibrator,
            PageHinkley, ControlChart, EvolutionaryMLSystem,
        )
        all_passed &= print_status("Core classes import", True)
    except ImportError as e:
        all_passed &= print_status("Core classes import", False)
        print(f"  Error: {e}")
        return False
    
    try:
        from evolutionary_ml_engine import make_genome, gb_factory, synthesize
        import numpy as np
        
        rng = np.random.default_rng(42)
        genome = make_genome(rng)
        all_passed &= print_status("HPGenome creation", True)
    except Exception as e:
        all_passed &= print_status("HPGenome creation", False)
        print(f"  Error: {e}")
    
    try:
        hp = genome.sample()
        all_passed &= print_status("Hyperparameter sampling", True)
    except Exception as e:
        all_passed &= print_status("Hyperparameter sampling", False)
        print(f"  Error: {e}")
    
    try:
        model = gb_factory(hp)
        all_passed &= print_status("Model factory", True)
    except Exception as e:
        all_passed &= print_status("Model factory", False)
        print(f"  Error: {e}")
    
    try:
        X, y = synthesize(n=100, drift_strength=0.0, seed=0)
        all_passed &= print_status("Data synthesis", True)
    except Exception as e:
        all_passed &= print_status("Data synthesis", False)
        print(f"  Error: {e}")
    
    return all_passed


def test_quick_run():
    """Test a quick run of the system."""
    print_header("Testing Quick System Run")
    
    try:
        from evolutionary_ml_engine import (
            EvolutionaryMLSystem, make_genome, gb_factory, synthesize
        )
        import numpy as np
        import random
        
        random.seed(42)
        rng = np.random.default_rng(42)
        genome = make_genome(rng)
        
        system = EvolutionaryMLSystem(
            model_factory=gb_factory,
            genome=genome,
            population_size=4,  # Small for speed
            alpha=0.1,
        )
        
        X0, y0 = synthesize(n=200, drift_strength=0.0, seed=0)
        system.seed(X0, y0, generations=1)  # Just 1 generation for speed
        
        print_status("System initialization and seeding", True)
        
        X1, y1 = synthesize(n=100, drift_strength=0.5, seed=1)
        result = system.step(X1, y1)
        
        print_status("Streaming cycle execution", True)
        
        # Verify expected keys in result
        expected_keys = ["auc", "conformal_coverage", "out_of_control", "action"]
        has_keys = all(k in result for k in expected_keys)
        print_status("Result structure valid", has_keys)
        
        return has_keys
        
    except Exception as e:
        print_status("Quick system run", False)
        print(f"  Error: {e}")
        return False


def test_visualizations():
    """Test that visualization modules load."""
    print_header("Testing Visualization Modules")
    
    all_passed = True
    
    if Path("visualize.py").exists():
        try:
            import visualize
            all_passed &= print_status("visualize.py loads", True)
        except Exception as e:
            all_passed &= print_status("visualize.py loads", False)
            print(f"  Error: {e}")
    else:
        all_passed &= print_status("visualize.py exists", False)
    
    if Path("dashboard.py").exists():
        try:
            # Don't actually import streamlit-dependent code, just check file exists
            with open("dashboard.py") as f:
                content = f.read()
                has_streamlit = "import streamlit" in content
            all_passed &= print_status("dashboard.py structure", has_streamlit)
        except Exception as e:
            all_passed &= print_status("dashboard.py structure", False)
            print(f"  Error: {e}")
    else:
        all_passed &= print_status("dashboard.py exists", False)
    
    return all_passed


def test_examples():
    """Test that example files exist and are valid."""
    print_header("Testing Example Use Cases")
    
    all_passed = True
    
    examples = [
        "examples/fraud_detection.py",
        "examples/churn_prediction.py",
    ]
    
    for example in examples:
        if Path(example).exists():
            try:
                with open(example) as f:
                    content = f.read()
                    # Check for key imports
                    has_imports = (
                        "from evolutionary_ml_engine import" in content
                        or "import evolutionary_ml_engine" in content
                    )
                all_passed &= print_status(f"{example} structure", has_imports)
            except Exception as e:
                all_passed &= print_status(f"{example} structure", False)
                print(f"  Error: {e}")
        else:
            all_passed &= print_status(f"{example} exists", False)
    
    return all_passed


def test_documentation():
    """Test that all required documentation exists."""
    print_header("Testing Documentation Files")
    
    all_passed = True
    
    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        ".gitignore",
        "QUICKSTART.md",
    ]
    
    for filename in required_files:
        exists = Path(filename).exists()
        all_passed &= print_status(f"{filename} exists", exists)
        
        if exists and filename in ["README.md", "QUICKSTART.md"]:
            with open(filename) as f:
                content = f.read()
                has_content = len(content) > 100
                all_passed &= print_status(f"{filename} has content", has_content)
    
    return all_passed


def main():
    """Run all verification tests."""
    print_header("EVOLUTIONARY ML ENGINE - VERIFICATION SUITE")
    print("Testing all components before hackathon submission...")
    
    all_tests_passed = True
    
    # Run all test suites
    all_tests_passed &= test_imports()
    all_tests_passed &= test_documentation()
    all_tests_passed &= test_core_engine()
    all_tests_passed &= test_quick_run()
    all_tests_passed &= test_visualizations()
    all_tests_passed &= test_examples()
    
    # Final summary
    print_header("VERIFICATION SUMMARY")
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print()
        print("Your Evolutionary ML Engine is ready for hackathon submission!")
        print()
        print("Next steps:")
        print("  1. Review SUBMISSION_CHECKLIST.md")
        print("  2. Create your demo video")
        print("  3. Update README.md with your contact info")
        print("  4. Submit to DevPost!")
        print()
        return 0
    else:
        print("⚠️  SOME TESTS FAILED ⚠️")
        print()
        print("Please review the errors above and fix any issues.")
        print("Re-run this script after making corrections.")
        print()
        print("Common fixes:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Ensure all files are in the correct locations")
        print("  - Check Python version: python --version (need 3.10+)")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
