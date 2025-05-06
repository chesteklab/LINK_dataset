import os
import shutil
import pandas as pd
import config

def main():
    review_csv = config.resultspath
    src_dir = config.preprocessingpath
    dest_dir = os.path.join(config.outpath, 'only_good_days')
    os.makedirs(dest_dir, exist_ok=True)

    df = pd.read_csv(review_csv)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    good_dates = df.loc[df['Status'] == 'good', 'Date'].unique()
    print(f"Found {len(good_dates)} good days. Copying .pkl files...")

    for date_str in good_dates:
        fname    = f"{date_str}_preprocess.pkl"
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dest_dir, fname)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {fname}")
        else:
            print(f"Not copied: {fname} not found in {config.preprocessingpath}")
    
    print("Done!")

if __name__ == "__main__":
    main()

