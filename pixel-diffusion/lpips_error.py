import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import sys
import os
import re
import glob
import lpips

def load_images_from_folder(path, transform=None):
    """Load images from a flat folder."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if transform is None:
        # Default transform: convert to tensor and normalize to [-1,1] for LPIPS
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
        ])
    
    # Get all PNG files in the folder
    png_files = glob.glob(os.path.join(path, "*.png"))
    
    if not png_files:
        raise ValueError(f"No PNG files found in {path}")
    
    images = []
    filepaths = []
    
    for img_path in sorted(png_files):
        try:
            img = Image.open(img_path)
            img = transform(img)
            images.append(img)
            filepaths.append(img_path)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    
    return images, filepaths

def get_image_mapping(images, filepaths):
    """
    Create mapping from base index to image data for pattern matching.
    Expects filenames like idx.png and idx_a.png where a can be any number.
    """
    mapping = {}
    
    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        
        # Extract base index from filename
        # Pattern: idx.png or idx_a.png where idx and a are numbers
        match = re.match(r'^(\d+)(?:_\d+)?\.png$', filename)
        if match:
            base_idx = int(match.group(1))
            mapping[base_idx] = i
        else:
            print(f"Warning: Filename {filename} doesn't match expected pattern")
    
    return mapping

def compute_lpips_differences(original_images, original_filepaths, restored_images, restored_filepaths, lpips_model):
    """
    Compute LPIPS differences between corresponding images in two datasets.
    Matches images based on filename patterns: idx.png and idx_a.png
    """
    # Get filename mappings
    original_mapping = get_image_mapping(original_images, original_filepaths)
    restored_mapping = get_image_mapping(restored_images, restored_filepaths)
    
    # Find common indices
    common_indices = set(original_mapping.keys()) & set(restored_mapping.keys())
    
    if not common_indices:
        raise ValueError("No matching image pairs found based on filename patterns")
    
    print(f"Found {len(common_indices)} matching image pairs")
    
    lpips_differences = []
    sorted_indices = sorted(common_indices)
    
    print(f"Computing LPIPS differences for {len(sorted_indices)} image pairs...")
    
    with torch.no_grad():
        for base_idx in sorted_indices:
            # Get image indices for this base index
            orig_idx = original_mapping[base_idx]
            rest_idx = restored_mapping[base_idx]
            
            # Get images
            orig_img = original_images[orig_idx]
            rest_img = restored_images[rest_idx]
            
            # Assert same shape (no resizing)
            assert orig_img.shape == rest_img.shape, f"Image shape mismatch for index {base_idx}: {orig_img.shape} vs {rest_img.shape}"
            
            # Add batch dimension for LPIPS
            orig_img_batch = orig_img.unsqueeze(0)
            rest_img_batch = rest_img.unsqueeze(0)
            
            # Compute LPIPS distance
            lpips_dist = lpips_model(orig_img_batch.to('cuda'), rest_img_batch.to('cuda'))
            lpips_differences.append(lpips_dist.item())
    
    return lpips_differences

def main():
    parser = argparse.ArgumentParser(description='Compute LPIPS differences between original and restored image datasets')
    parser.add_argument('original_dataset', type=str, help='Path to original image dataset')
    parser.add_argument('restored_dataset', type=str, help='Path to restored image dataset')
    parser.add_argument('--network', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'], 
                       help='Network to use for LPIPS (default: alex)')
    
    args = parser.parse_args()
    
    # Print arguments
    print("Arguments:")
    print(f"original_dataset: {args.original_dataset}")
    print(f"restored_dataset: {args.restored_dataset}")
    print(f"network: {args.network}")
    print()
    
    # Set up LPIPS model
    print(f"Loading LPIPS model with {args.network} network...")
    lpips_model = lpips.LPIPS(net=args.network).to('cuda')
    
    # Set up transforms (LPIPS expects [-1,1] normalized images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        # Load datasets
        print(f"Loading original images from: {args.original_dataset}")
        original_images, original_filepaths = load_images_from_folder(args.original_dataset, transform)
        print(f"Original images loaded: {len(original_images)} images")
        
        print(f"Loading restored images from: {args.restored_dataset}")
        restored_images, restored_filepaths = load_images_from_folder(args.restored_dataset, transform)
        print(f"Restored images loaded: {len(restored_images)} images")
        
        # Assert same dataset sizes
        assert len(original_images) == len(restored_images), \
            f"Dataset sizes must be equal. Original: {len(original_images)}, Restored: {len(restored_images)}"
        
        # Compute LPIPS differences
        lpips_diffs = compute_lpips_differences(original_images, original_filepaths, restored_images, restored_filepaths, lpips_model)
        
        if not lpips_diffs:
            print("No valid image pairs found for comparison.")
            return
        
        # Calculate statistics
        avg_lpips = sum(lpips_diffs) / len(lpips_diffs)
        min_lpips = min(lpips_diffs)
        max_lpips = max(lpips_diffs)
        
        # Print results
        print(f"Results:")
        print(f"Number of image pairs compared: {len(lpips_diffs)}")
        print(f"Average LPIPS distance: {avg_lpips:.6f}")
        print(f"Minimum LPIPS distance: {min_lpips:.6f}")
        print(f"Maximum LPIPS distance: {max_lpips:.6f}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()