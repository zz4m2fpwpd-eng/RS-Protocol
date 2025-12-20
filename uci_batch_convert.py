#!/usr/bin/env python3
"""
UCI .data File Batch Converter
Converts ALL .data files in a directory to CSV format for TITAN RS
"""

import os
import pandas as pd
from pathlib import Path
import sys

def convert_all_data_files(input_dir, output_dir=None):
    """Convert all .data files to .csv"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory not found: {input_dir}")
        return
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    data_files = list(input_path.glob("*.data"))
    
    if not data_files:
        print(f"⚠️  No .data files found in {input_dir}")
        return
    
    print(f"📂 Found {len(data_files)} .data files")
    print(f"📁 Output: {output_path}")
    print()
    
    converted_count = 0
    
    for data_file in data_files:
        try:
            # Read .data file (no headers assumed)
            df = pd.read_csv(str(data_file), header=None, sep='[,\t\s]+', engine='python')
            
            # Remove rows with '?' (missing values in UCI format)
            df = df.replace('?', pd.NA).dropna()
            
            # Save to CSV
            csv_filename = data_file.stem + ".csv"
            csv_path = output_path / csv_filename
            df.to_csv(str(csv_path), index=False)
            
            print(f"✅ {data_file.name}")
            print(f"   → {csv_filename} ({len(df)} rows × {len(df.columns)} cols)")
            converted_count += 1
            
        except Exception as e:
            print(f"❌ {data_file.name}: {str(e)}")
    
    print()
    print(f"✅ Converted {converted_count}/{len(data_files)} files")
    print(f"📍 CSV files ready in: {output_path}")
    print()
    print("Next: Run TITAN RS on all CSV files")
    print(f"  for csv in {output_path}/*.csv; do python3 titan_rs.py \"$csv\"; done")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 uci_batch_convert.py <input_dir> [output_dir]")
        print()
        print("Example:")
        print("  python3 uci_batch_convert.py ~/uci_datasets ~/uci_datasets_csv")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_all_data_files(input_dir, output_dir)
