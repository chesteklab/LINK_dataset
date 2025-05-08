import os
import shutil
import pandas as pd
import config
from tqdm import tqdm

def main():
    review_csv = config.reviewpath
    src_dir = config.preprocessingdir
    dest_dir = config.good_daysdir
    os.makedirs(dest_dir, exist_ok=True)

    df = pd.read_csv(review_csv)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    good_dates = df.loc[df['Status'] == 'good', 'Date'].unique()
    print(f"Found {len(good_dates)} good days. Copying .pkl files...")

    for date_str in tqdm(good_dates):
        fname    = f"{date_str}_preprocess.pkl"
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dest_dir, fname)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {fname}")
        else:
            print(f"Not copied: {fname} not found in {config.preprocessingdir}")
    
    print("Done!")

if __name__ == "__main__":
    main()

