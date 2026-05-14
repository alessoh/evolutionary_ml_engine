"""
Fix encoding issues on Windows by re-saving files with UTF-8 encoding.
Run this to fix the charmap codec errors in verify.py
"""

import os
from pathlib import Path

def fix_file_encoding(filepath):
    """Read and re-save file with UTF-8 encoding."""
    try:
        # Try reading with UTF-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Write back with UTF-8 explicitly
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        print(f"✅ Fixed: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    print("="*80)
    print("FIXING FILE ENCODING ISSUES")
    print("="*80)
    print()
    
    # Files that need fixing
    files_to_fix = [
        'dashboard.py',
        'examples/fraud_detection.py',
        'examples/churn_prediction.py',
    ]
    
    fixed_count = 0
    
    for filepath in files_to_fix:
        if Path(filepath).exists():
            if fix_file_encoding(filepath):
                fixed_count += 1
        else:
            print(f"⚠️  File not found: {filepath}")
    
    print()
    print("="*80)
    print(f"Fixed {fixed_count}/{len(files_to_fix)} files")
    print("="*80)
    print()
    print("Now run: python verify.py")
    print()

if __name__ == "__main__":
    main()
