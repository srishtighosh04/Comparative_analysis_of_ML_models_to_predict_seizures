#!/usr/bin/env python3
"""
Main runner script for EEG seizure detection analysis
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import numpy
        import matplotlib
        import seaborn
        import scipy
        import sklearn
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False

def run_analysis():
    """Run the main analysis script"""
    try:
        print("🔬 Running EEG Seizure Detection Analysis...")
        print("=" * 50)
        
        # Import and run main analysis
        analysis_path = project_root / "src" / "main_analysis.py"
        if analysis_path.exists():
            exec(open(analysis_path).read())
        else:
            print(f"❌ Analysis script not found: {analysis_path}")
            return False
        
        print("=" * 50)
        print("✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

def launch_notebook():
    """Launch Jupyter notebook"""
    try:
        notebooks_dir = project_root / "notebooks"
        if notebooks_dir.exists():
            print("🚀 Launching Jupyter notebook...")
            print(f"📂 Opening directory: {notebooks_dir}")
            
            # Try to launch Jupyter
            result = subprocess.run([
                sys.executable, "-m", "jupyter", "notebook", str(notebooks_dir)
            ], check=False)
            
            if result.returncode != 0:
                print("❌ Failed to launch Jupyter notebook")
                print("Make sure Jupyter is installed: pip install jupyter")
                return False
        else:
            print(f"❌ Notebooks directory not found: {notebooks_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error launching notebook: {e}")
        return False

def show_dataset_info():
    """Show information about available datasets"""
    try:
        from src.utils.data_loader import list_available_datasets, get_dataset_info
        from config.config import RAW_DATA_DIR
        
        print("📊 Available Datasets:")
        print("=" * 50)
        
        datasets = list_available_datasets()
        
        for category, files in datasets.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            if files:
                for file in files:
                    print(f"  ✓ {file}")
            else:
                print("  ❌ No files found")
        
        # Show data directory info
        print(f"\n📁 Data Directory: {RAW_DATA_DIR}")
        if os.path.exists(RAW_DATA_DIR):
            npz_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.npz')]
            print(f"   Total .npz files: {len(npz_files)}")
            
            total_size = sum(os.path.getsize(os.path.join(RAW_DATA_DIR, f)) 
                           for f in npz_files) / (1024**3)  # GB
            print(f"   Total size: {total_size:.2f} GB")
        else:
            print("   ❌ Directory not found")
        
    except Exception as e:
        print(f"❌ Error getting dataset info: {e}")

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(
        description="EEG Seizure Detection Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --analysis          # Run main analysis
  python run_analysis.py --notebook          # Launch Jupyter notebook
  python run_analysis.py --info              # Show dataset information
  python run_analysis.py --check             # Check dependencies
        """
    )
    
    parser.add_argument("--analysis", action="store_true",
                       help="Run main analysis script")
    parser.add_argument("--notebook", action="store_true",
                       help="Launch Jupyter notebook")
    parser.add_argument("--info", action="store_true",
                       help="Show dataset information")
    parser.add_argument("--check", action="store_true",
                       help="Check dependencies")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("🧠 EEG Seizure Detection Project")
    print("=" * 50)
    
    if args.check:
        check_dependencies()
    elif args.info:
        show_dataset_info()
    elif args.notebook:
        if check_dependencies():
            launch_notebook()
    elif args.analysis:
        if check_dependencies():
            run_analysis()

if __name__ == "__main__":
    main()
