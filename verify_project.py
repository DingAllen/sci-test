#!/usr/bin/env python3
"""
Verification script to validate all project deliverables.
"""

import os
import json
import pickle

def verify_project():
    """Verify all project components are present and valid."""
    
    print("="*80)
    print("PROJECT VERIFICATION SCRIPT")
    print("="*80)
    
    errors = []
    warnings = []
    
    # Check source code files
    print("\n[1/8] Checking source code files...")
    source_files = [
        'src/test_functions.py',
        'src/optimizers.py', 
        'src/experiment_runner.py',
        'src/visualization.py',
        'run_experiments.py'
    ]
    
    for f in source_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✓ {f} ({size} bytes)")
        else:
            errors.append(f"Missing source file: {f}")
            print(f"  ✗ {f} MISSING")
    
    # Check documentation
    print("\n[2/8] Checking documentation...")
    docs = [
        'README.md',
        'EXPERIMENT_LOG.md',
        'COMPLETION_SUMMARY.md',
        'experiment_plan.md',
        'agent_task.md'
    ]
    
    for f in docs:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✓ {f} ({size} bytes)")
        else:
            errors.append(f"Missing documentation: {f}")
            print(f"  ✗ {f} MISSING")
    
    # Check paper
    print("\n[3/8] Checking LaTeX paper...")
    if os.path.exists('paper/paper.tex'):
        tex_size = os.path.getsize('paper/paper.tex')
        print(f"  ✓ paper/paper.tex ({tex_size} bytes)")
    else:
        errors.append("Missing LaTeX source: paper/paper.tex")
        print(f"  ✗ paper/paper.tex MISSING")
    
    if os.path.exists('paper/paper.pdf'):
        pdf_size = os.path.getsize('paper/paper.pdf')
        print(f"  ✓ paper/paper.pdf ({pdf_size / 1024 / 1024:.1f} MB)")
        if pdf_size < 1000000:  # Less than 1MB might indicate issue
            warnings.append("PDF file seems small, may be incomplete")
    else:
        errors.append("Missing PDF: paper/paper.pdf")
        print(f"  ✗ paper/paper.pdf MISSING")
    
    # Check figures
    print("\n[4/8] Checking generated figures...")
    figures_dir = 'figures'
    if os.path.exists(figures_dir):
        png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"  ✓ Found {len(png_files)} PNG figures")
        
        expected_figures = 22
        if len(png_files) == expected_figures:
            print(f"  ✓ Correct number of figures ({expected_figures})")
        else:
            warnings.append(f"Expected {expected_figures} figures, found {len(png_files)}")
            print(f"  ⚠ Expected {expected_figures} figures, found {len(png_files)}")
        
        total_size = sum(os.path.getsize(os.path.join(figures_dir, f)) for f in png_files)
        print(f"  ✓ Total figures size: {total_size / 1024 / 1024:.1f} MB")
    else:
        errors.append("Missing figures directory")
        print(f"  ✗ figures/ directory MISSING")
    
    # Check results
    print("\n[5/8] Checking experimental results...")
    results_dir = 'results'
    if os.path.exists(results_dir):
        pkl_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        
        print(f"  ✓ Found {len(pkl_files)} result files (.pkl)")
        print(f"  ✓ Found {len(json_files)} statistics files (.json)")
        
        total_size = sum(os.path.getsize(os.path.join(results_dir, f)) 
                        for f in os.listdir(results_dir))
        print(f"  ✓ Total results size: {total_size / 1024 / 1024:.1f} MB")
        
        # Validate a sample JSON file
        if json_files:
            sample_json = os.path.join(results_dir, json_files[0])
            try:
                with open(sample_json, 'r') as f:
                    data = json.load(f)
                    required_keys = ['mean_final_loss', 'std_final_loss', 'success_rate']
                    if all(k in data for k in required_keys):
                        print(f"  ✓ JSON format validated")
                    else:
                        warnings.append("JSON files missing required keys")
            except Exception as e:
                warnings.append(f"Error reading JSON: {e}")
    else:
        errors.append("Missing results directory")
        print(f"  ✗ results/ directory MISSING")
    
    # Check requirements
    print("\n[6/8] Checking requirements...")
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            reqs = f.readlines()
            print(f"  ✓ requirements.txt ({len(reqs)} packages)")
    else:
        warnings.append("Missing requirements.txt")
        print(f"  ⚠ requirements.txt MISSING")
    
    # Check .gitignore
    print("\n[7/8] Checking repository configuration...")
    if os.path.exists('.gitignore'):
        print(f"  ✓ .gitignore present")
    else:
        warnings.append("Missing .gitignore")
        print(f"  ⚠ .gitignore MISSING")
    
    # Count total deliverables
    print("\n[8/8] Counting total deliverables...")
    total_files = sum(len(files) for _, _, files in os.walk('.'))
    git_files = sum(len(files) for _, _, files in os.walk('.git'))
    project_files = total_files - git_files
    print(f"  ✓ Total project files: {project_files}")
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    
    if not errors and not warnings:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nProject is complete and ready for submission.")
        return True
    elif errors:
        print("\n❌ VERIFICATION FAILED")
        print("Please fix errors before submission.")
        return False
    else:
        print("\n✅ VERIFICATION PASSED WITH WARNINGS")
        print("Project is acceptable but could be improved.")
        return True


if __name__ == '__main__':
    success = verify_project()
    exit(0 if success else 1)
