import pandas as pd

# Path to your CSV
CSV_PATH = "/home/vjti-comp/WEEDSBL/scripts/sota/dataset_utils/mask_class_percentages.csv"

# Output text file
OUTPUT_PATH = "/home/vjti-comp/WEEDSBL/scripts/sota/dataset_utils/5000_mixed_images.txt"

def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Expect these columns:
    required_cols = {"clearfilename", "crop_percentage", "weed_percentage"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}, found: {df.columns.tolist()}")

    # Sort by weed_percentage (descending) and take top 2500
    df_sorted = df.sort_values("weed_percentage", ascending=False)
    top_2500_weed = df_sorted.head(2500)
    
    # Filter rows with 0 weed percentage, then sort by crop percentage
    df_crop_filtered = df[df["weed_percentage"] == 0]
    df_crop_sorted = df_crop_filtered.sort_values("crop_percentage", ascending=False)
    top_2500_crop = df_crop_sorted.head(2500)
    
    # Combine both datasets
    new_df = pd.concat([top_2500_weed, top_2500_crop])

    # Write filenames to text file, one per line
    with open(OUTPUT_PATH, "w") as f:
        for filename in new_df["clearfilename"]:
            f.write(f"{filename}\n")

    print(f"Saved {len(new_df)} image names to {OUTPUT_PATH}")
    print(f"  - {len(top_2500_weed)} images with highest weed percentage")
    print(f"  - {len(top_2500_crop)} images with highest crop percentage (0% weed)")

if __name__ == "__main__":
    main()
