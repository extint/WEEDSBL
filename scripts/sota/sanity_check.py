import os
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

def analyze_mask_percentages(mask_folder, output_csv='mask_class_percentages.csv'):
    """
    Analyze mask images and calculate percentage of background, crop, and weed pixels.
    """
    mask_folder = Path(mask_folder)
    
    if not mask_folder.exists():
        raise ValueError(f"Folder {mask_folder} does not exist")
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = [f for f in mask_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    results = []
    total_images = len(image_files)
    
    print(f"Processing {total_images} images from {mask_folder}...")
    print(f"Expected image dimensions: 1296 x 966")
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Expected total pixels (1296 * 966 = 1,251,936)
            expected_pixels = 1296 * 966
            
            # Count pixels for each class
            background_pixels = np.sum(img_array == 0)
            crop_pixels = np.sum(img_array == 1)
            weed_pixels = np.sum(img_array == 2)
            
            # Calculate percentages
            background_percentage = (background_pixels / expected_pixels) * 100
            crop_percentage = (crop_pixels / expected_pixels) * 100
            weed_percentage = (weed_pixels / expected_pixels) * 100
            
            results.append({
                'filename': img_path.name,
                'background_percentage': background_percentage,
                'crop_percentage': crop_percentage,
                'weed_percentage': weed_percentage
            })
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{total_images} images...")
        
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Total images processed: {len(results)}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Background - Mean: {df['background_percentage'].mean():.2f}%")
    print(f"Crop - Mean: {df['crop_percentage'].mean():.2f}%")
    print(f"Weed - Mean: {df['weed_percentage'].mean():.2f}%")
    
    return df


if __name__ == "__main__":
    MASK_FOLDER = "/home/vjti-comp/Downloads/FINAL_SUGARBEETS_DATASET/masks"
    df = analyze_mask_percentages(MASK_FOLDER, output_csv='mask_class_percentages.csv')


# import os
# import numpy as np
# from PIL import Image
# from pathlib import Path

# def count_images_without_weeds(mask_folder, threshold_pixels=None, threshold_percentage=None):
#     """
#     Count mask images with weed pixels below a specified threshold.
    
#     Args:
#         mask_folder: Path to folder containing mask images
#         threshold_pixels: Absolute number of weed pixels (e.g., 100 pixels)
#         threshold_percentage: Percentage of total pixels (e.g., 0.5 for 0.5%)
#     """
#     mask_folder = Path(mask_folder)
    
#     if not mask_folder.exists():
#         raise ValueError(f"Folder {mask_folder} does not exist")
    
#     image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
#     image_files = [f for f in mask_folder.iterdir() 
#                    if f.suffix.lower() in image_extensions]
    
#     images_without_weeds = []
#     weed_pixel_stats = []
#     total_images = len(image_files)
    
#     print(f"Processing {total_images} images from {mask_folder}...")
#     if threshold_pixels:
#         print(f"Threshold: {threshold_pixels} pixels")
#     elif threshold_percentage:
#         print(f"Threshold: {threshold_percentage}% of total pixels")
    
#     for img_path in image_files:
#         try:
#             img = Image.open(img_path)
#             img_array = np.array(img)
            
#             # Count weed pixels (value 2)
#             weed_pixel_count = np.sum(img_array == 2)
#             total_pixels = img_array.size
#             weed_percentage = (weed_pixel_count / total_pixels) * 100
            
#             # Determine threshold
#             if threshold_pixels is not None:
#                 below_threshold = weed_pixel_count < threshold_pixels
#             elif threshold_percentage is not None:
#                 below_threshold = weed_percentage < threshold_percentage
#             else:
#                 below_threshold = weed_pixel_count == 0
            
#             # Store statistics
#             weed_pixel_stats.append({
#                 'filename': img_path.name,
#                 'weed_pixels': weed_pixel_count,
#                 'total_pixels': total_pixels,
#                 'percentage': weed_percentage,
#                 'below_threshold': below_threshold
#             })
            
#             if below_threshold:
#                 images_without_weeds.append(img_path.name)
        
#         except Exception as e:
#             print(f"Error processing {img_path.name}: {e}")
    
#     count_without_weeds = len(images_without_weeds)
    
#     # Print results
#     print(f"\n{'='*60}")
#     print(f"Total images processed: {total_images}")
#     print(f"Images below weed threshold: {count_without_weeds}")
#     print(f"Images above weed threshold: {total_images - count_without_weeds}")
#     print(f"Percentage below threshold: {(count_without_weeds/total_images)*100:.2f}%")
#     print(f"{'='*60}")
    
#     # Calculate statistics
#     if weed_pixel_stats:
#         weed_counts = [s['weed_pixels'] for s in weed_pixel_stats if not s['below_threshold']]
#         if weed_counts:
#             print(f"\nWeed pixel statistics (for images above threshold):")
#             print(f"  Min weed pixels: {min(weed_counts)}")
#             print(f"  Max weed pixels: {max(weed_counts)}")
#             print(f"  Mean weed pixels: {np.mean(weed_counts):.2f}")
#             print(f"  Median weed pixels: {np.median(weed_counts):.2f}")
    
#     if images_without_weeds:
#         print(f"\nImages below weed threshold (first 20):")
#         for filename in images_without_weeds[:20]:
#             stats = next(s for s in weed_pixel_stats if s['filename'] == filename)
#             print(f"  - {filename}: {stats['weed_pixels']} pixels ({stats['percentage']:.3f}%)")
    
#     return count_without_weeds, total_images, images_without_weeds, weed_pixel_stats

# MASK_FOLDER = "/home/vjti-comp/Downloads/FINAL_SUGARBEETS_DATASET/masks"

# # Option 1: Absolute pixel threshold (recommended for small weeds/noise)
# count, total, filenames, stats = count_images_without_weeds(
#     MASK_FOLDER, 
#     threshold_pixels=200  # Ignore images with <100 weed pixels
# )

# print("Count of without weeds is:",count)
# print("Total count is:", total)