#!/usr/bin/env python3
"""
Diagnostic script to test import issues and identify runtime problems
"""

import sys
import traceback
from typing import Dict, List, Any

def test_basic_imports() -> Dict[str, Any]:
    """Test basic Python imports"""
    results = {}
    
    try:
        import torch
        results['torch'] = {'status': 'success', 'version': torch.__version__}
    except ImportError as e:
        results['torch'] = {'status': 'error', 'error': str(e)}
    
    try:
        import torchvision
        results['torchvision'] = {'status': 'success', 'version': torchvision.__version__}
    except ImportError as e:
        results['torchvision'] = {'status': 'error', 'error': str(e)}
    
    try:
        import numpy
        results['numpy'] = {'status': 'success', 'version': numpy.__version__}
    except ImportError as e:
        results['numpy'] = {'status': 'error', 'error': str(e)}
    
    return results

def test_optional_backends() -> Dict[str, Any]:
    """Test optional SNN backend imports"""
    results = {}
    
    try:
        import spikingjelly
        results['spikingjelly'] = {'status': 'success', 'version': getattr(spikingjelly, '__version__', 'unknown')}
    except ImportError as e:
        results['spikingjelly'] = {'status': 'error', 'error': str(e)}
    
    try:
        import snntorch
        results['snntorch'] = {'status': 'success', 'version': getattr(snntorch, '__version__', 'unknown')}
    except ImportError as e:
        results['snntorch'] = {'status': 'error', 'error': str(e)}
    
    try:
        import brian2
        results['brian2'] = {'status': 'success', 'version': getattr(brian2, '__version__', 'unknown')}
    except ImportError as e:
        results['brian2'] = {'status': 'error', 'error': str(e)}
    
    return results

def test_toolkit_imports() -> Dict[str, Any]:
    """Test toolkit module imports individually"""
    results = {}
    
    # Test individual modules first
    modules_to_test = [
        'toolkit.config',
        'toolkit.logging_utils', 
        'toolkit.data_processing',
        'toolkit.core',
        'toolkit.utils',
        'toolkit.training',
        'toolkit.tuning',
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            results[module_name] = {'status': 'success'}
        except Exception as e:
            results[module_name] = {'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test main toolkit import
    try:
        import toolkit
        results['toolkit'] = {'status': 'success'}
    except Exception as e:
        results['toolkit'] = {'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}
    
    return results

def test_example_imports() -> Dict[str, Any]:
    """Test example script imports"""
    results = {}
    
    try:
        sys.path.insert(0, '.')  # Add current directory to path
        from examples import mnist_classifier
        results['mnist_classifier'] = {'status': 'success'}
    except Exception as e:
        results['mnist_classifier'] = {'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}
    
    return results

def main():
    """Run all diagnostic tests"""
    print("=" * 60)
    print("NEUROMORPHIC SNN TOOLKIT - IMPORT DIAGNOSTICS")
    print("=" * 60)
    
    print("\n1. Testing Basic Dependencies...")
    basic_results = test_basic_imports()
    for name, result in basic_results.items():
        status = result['status']
        if status == 'success':
            version = result.get('version', 'unknown')
            print(f"  ✓ {name}: {version}")
        else:
            print(f"  ✗ {name}: {result['error']}")
    
    print("\n2. Testing Optional SNN Backends...")
    backend_results = test_optional_backends()
    for name, result in backend_results.items():
        status = result['status']
        if status == 'success':
            version = result.get('version', 'unknown')
            print(f"  ✓ {name}: {version}")
        else:
            print(f"  ✗ {name}: {result['error']}")
    
    print("\n3. Testing Toolkit Modules...")
    toolkit_results = test_toolkit_imports()
    for name, result in toolkit_results.items():
        status = result['status']
        if status == 'success':
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}: {result['error']}")
            if 'traceback' in result:
                print(f"    Traceback: {result['traceback'][:200]}...")
    
    print("\n4. Testing Example Scripts...")
    example_results = test_example_imports()
    for name, result in example_results.items():
        status = result['status']
        if status == 'success':
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}: {result['error']}")
            if 'traceback' in result:
                print(f"    Traceback: {result['traceback'][:200]}...")
    
    # Summary
    all_results = {**basic_results, **backend_results, **toolkit_results, **example_results}
    failed_imports = [name for name, result in all_results.items() if result['status'] == 'error']
    
    print(f"\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total modules tested: {len(all_results)}")
    print(f"Failed imports: {len(failed_imports)}")
    
    if failed_imports:
        print(f"Failed modules: {', '.join(failed_imports)}")
        print("\nRECOMMENDED ACTIONS:")
        
        if 'torch' in failed_imports:
            print("  1. Install PyTorch: pip install torch torchvision")
        if 'spikingjelly' in failed_imports:
            print("  2. Install SpikingJelly: pip install spikingjelly")
        if 'snntorch' in failed_imports:
            print("  3. Install snnTorch: pip install snntorch")
        if any('toolkit' in name for name in failed_imports):
            print("  4. Check for circular imports in toolkit modules")
            print("  5. Verify all toolkit files exist and are syntactically correct")
    else:
        print("All imports successful!")

if __name__ == "__main__":
    main()