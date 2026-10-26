import pandas as pd
from pathlib import Path

models = ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x']
summary = []

for m in models:
    csv_path = Path(f'kaggle_metrics/YOLO_object-detection/backup/{m}/results.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Find best epoch by mAP50
        map50_col = [c for c in df.columns if 'mAP50(B)' in c][0]
        map50_95_col = [c for c in df.columns if 'mAP50-95(B)' in c][0]
        p_col = [c for c in df.columns if 'precision(B)' in c][0]
        r_col = [c for c in df.columns if 'recall(B)' in c][0]
        
        best_idx = df[map50_col].idxmax()
        best_row = df.loc[best_idx]
        last_row = df.iloc[-1]
        
        summary.append({
            'Model': m,
            'Epochs': len(df),
            'Best_Epoch': int(best_row['epoch']),
            'Val_Precision': round(float(best_row[p_col]), 4),
            'Val_Recall': round(float(best_row[r_col]), 4),
            'Val_mAP50': round(float(best_row[map50_col]), 4),
            'Val_mAP50_95': round(float(best_row[map50_95_col]), 4),
            'Train_Time_Min': round(float(last_row['time']) / 60.0, 2)
        })

sum_df = pd.DataFrame(summary)
print(sum_df.to_string(index=False))
