import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

def generate_synthetic_data(num_samples, output_dir, img_size=(256, 256)):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating synthetic data"):
        # Create empty image and mask
        image = np.zeros(img_size, dtype=np.uint8)
        mask = np.zeros(img_size, dtype=np.uint8)
        
        # Randomly place some "organs" (ellipses)
        num_organs = np.random.randint(1, 5)
        for _ in range(num_organs):
            center = (np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1]))
            axes = (np.random.randint(10, 50), np.random.randint(10, 50))
            angle = np.random.randint(0, 180)
            
            # Fill mask with class 1
            cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
            
            # Fill image with some intensity
            intensity = np.random.randint(100, 200)
            cv2.ellipse(image, center, axes, angle, 0, 360, intensity, -1)
            
        # Add noise to image
        noise = np.random.normal(0, 10, img_size).astype(np.uint8)
        image = cv2.add(image, noise)
        
        # Save files
        cv2.imwrite(os.path.join(output_dir, 'images', f'img_{i:04d}.png'), image)
        cv2.imwrite(os.path.join(output_dir, 'masks', f'mask_{i:04d}.png'), mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    args = parser.parse_args()
    
    generate_synthetic_data(args.num_samples, args.output_dir)
