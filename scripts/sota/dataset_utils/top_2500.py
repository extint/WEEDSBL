import pandas as pd

# Path to your CSV
CSV_PATH = "/home/vjti-comp/WEEDSBL/scripts/sota/mask_class_percentages.csv"

# Output text file
OUTPUT_PATH = "/home/vjti-comp/WEEDSBL/scripts/sota/top_2500_weed_images.txt"

def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Expect these columns:
    # 'clearfilename', 'background_percentage', 'crop_percentage', 'weed_percentage'
    required_cols = {"clearfilename", "weed_percentage"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}, found: {df.columns.tolist()}")

    # Sort by weed_percentage (descending) and take top 2500
    df_sorted = df.sort_values("weed_percentage", ascending=False)
    top_2500 = df_sorted.head(2500)

    # Write filenames to text file, one per line
    with open(OUTPUT_PATH, "w") as f:
        for filename in top_2500["clearfilename"]:
            f.write(f"{filename}\n")

    print(f"Saved {len(top_2500)} image names to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
