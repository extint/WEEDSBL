import pandas as pd

# Paths
CSV_PATH = "/home/vjti-comp/WEEDSBL/scripts/sota/dataset_utils/mask_class_percentages.csv"
IMAGE_LIST_PATH = "/home/vjti-comp/WEEDSBL/scripts/sota/dataset_utils/5000_mixed_images.txt"
OUTPUT_DIR = "/home/vjti-comp/Downloads/SUGARBEETS_MIXED_DATASET/splits/"

def create_stratified_splits_every_10th():
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    
    # Read filtered image list
    with open(IMAGE_LIST_PATH, 'r') as f:
        selected_images = [line.strip() for line in f.readlines()]
    
    # Filter to selected images
    df_filtered = df[df['clearfilename'].isin(selected_images)].copy()
    
    # CRITICAL: Sort by weed_percentage for stratification
    df_sorted = df_filtered.sort_values('weed_percentage').reset_index(drop=True)
    
    # Assign splits based on position in sorted list
    split_labels = []
    for idx in range(len(df_sorted)):
        position = idx % 10
        if position < 8:  # First 8 of every 10
            split_labels.append('train')
        elif position == 8:  # 9th image
            split_labels.append('test')
        else:  # 10th image
            split_labels.append('val')
    
    df_sorted['split'] = split_labels
    
    # Verify
    print(f"Total images: {len(df_sorted)}")
    for split in ['train', 'test', 'val']:
        count = (df_sorted['split'] == split).sum()
        pct = count / len(df_sorted) * 100
        split_data = df_sorted[df_sorted['split'] == split]
        print(f"\n{split.upper()}: {count} images ({pct:.1f}%)")
        print(f"  Weed % - Mean: {split_data['weed_percentage'].mean():.3f}, "
              f"Median: {split_data['weed_percentage'].median():.3f}")
    
    # Save to files
    for split in ['train', 'test', 'val']:
        split_data = df_sorted[df_sorted['split'] == split]
        output_path = f"{OUTPUT_DIR}{split}.txt"
        with open(output_path, 'w') as f:
            for filename in split_data['clearfilename']:
                f.write(f"{filename[5:-4]}\n")
        print(f"\n Saved {output_path}")

if __name__ == "__main__":
    create_stratified_splits_every_10th()
