"""
Add SMAPE to all existing results. 
Works with both *_preds.csv and *_metrics.csv files.
NO RETRAINING - only post-processing existing results.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def smape(y_true, y_pred, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error (0-100%)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def process_preds_file(csv_path):
    """Compute SMAPE from predictions file (date, y_true, y_pred)"""
    try:
        df = pd.read_csv(csv_path)
        
        if 'y_true' not in df.columns or 'y_pred' not in df.columns:
            return None
        
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        
        # Remove NaNs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return None
        
        # Compute SMAPE
        smape_val = smape(y_true, y_pred)
        
        # Extract metadata from path
        parts = csv_path. parts
        filename = csv_path.stem  # without . csv
        
        return {
            'file': csv_path. name,
            'path': str(csv_path),
            'smape': smape_val,
            'n_samples': len(y_true),
            'source': 'predictions'
        }
        
    except Exception as e: 
        print(f"❌ Error processing {csv_path}: {e}")
        return None

def process_metrics_file(csv_path):
    """Add SMAPE to existing metrics file (mae, rmse, r2)"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check if SMAPE already exists
        if 'smape' in df.columns:
            print(f"⏭️  {csv_path. name} already has SMAPE")
            return None
        
        # We need to find the corresponding _preds. csv file
        preds_path = csv_path.parent / csv_path.name.replace('_metrics.csv', '_preds.csv')
        
        if not preds_path.exists():
            print(f"⚠️  No predictions file found for {csv_path. name}")
            return None
        
        # Compute SMAPE from preds file
        preds_df = pd.read_csv(preds_path)
        
        if 'y_true' not in preds_df.columns or 'y_pred' not in preds_df. columns:
            return None
        
        y_true = preds_df['y_true'].values
        y_pred = preds_df['y_pred'].values
        
        # Remove NaNs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return None
        
        smape_val = smape(y_true, y_pred)
        
        # Add SMAPE to the metrics dataframe
        df['smape'] = smape_val
        
        # Save updated metrics file
        df.to_csv(csv_path, index=False)
        print(f"✅ Updated {csv_path.name} with SMAPE={smape_val:. 2f}%")
        
        # Return summary
        result = df.iloc[0].to_dict()
        result['file'] = csv_path.name
        result['path'] = str(csv_path)
        result['source'] = 'metrics_updated'
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {csv_path}: {e}")
        return None

def main():
    """Main execution"""
    search_dirs = ['results', 'notebooks/results']
    
    all_results = []
    
    print("="*70)
    print(" ADD SMAPE TO ALL EXISTING RESULTS")
    print("="*70)
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            print(f"⚠️  Directory not found: {search_dir}")
            continue
        
        print(f"\n🔍 Scanning:  {search_dir}")
        
        # Process metrics files (update them in-place)
        print("\n📊 Updating *_metrics.csv files...")
        metrics_files = list(Path(search_dir).rglob('*_metrics.csv'))
        print(f"   Found {len(metrics_files)} metrics files")
        
        for metrics_file in metrics_files:
            result = process_metrics_file(metrics_file)
            if result: 
                all_results.append(result)
        
        # Process standalone predictions files (for summary)
        print("\n📈 Processing *_preds.csv files...")
        preds_files = list(Path(search_dir).rglob('*_preds.csv'))
        print(f"   Found {len(preds_files)} prediction files")
        
        for preds_file in preds_files:
            result = process_preds_file(preds_file)
            if result:
                all_results.append(result)
    
    # Create summary
    print("\n" + "="*70)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Save consolidated summary
        output_file = 'results/consolidated_metrics_with_smape.csv'
        os.makedirs('results', exist_ok=True)
        summary_df.to_csv(output_file, index=False)
        
        print(f"✅ SUCCESS!")
        print(f"   Processed:  {len(all_results)} files")
        print(f"   Output: {output_file}")
        
        # Show statistics
        if 'smape' in summary_df.columns:
            print("\n📊 SMAPE STATISTICS:")
            print(summary_df['smape'].describe())
            
            print("\n🏆 BEST MODELS (Lowest SMAPE):")
            best = summary_df.nsmallest(10, 'smape')[['file', 'smape', 'mae', 'rmse', 'r2']]
            print(best. to_string(index=False))
            
            print("\n📉 WORST MODELS (Highest SMAPE):")
            worst = summary_df.nlargest(10, 'smape')[['file', 'smape', 'mae', 'rmse', 'r2']]
            print(worst.to_string(index=False))
    else:
        print("❌ No results processed")
    
    print("="*70)

if __name__ == '__main__':
    main()
