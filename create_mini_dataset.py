import os
import cv2
import numpy as np
try:
    from sklearn.datasets import fetch_lfw_people
except ImportError:
    print("Please install scikit-learn first: pip install scikit-learn")
    exit(1)

def generate_mini_dataset():
    print("Downloading 'Labeled Faces in the Wild' (A small, real-face dataset)...")
    print("This may take a minute to download (~30MB).")
    
    # Fetch a small dataset of real faces in color
    try:
        lfw_dataset = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return
        
    images = lfw_dataset.images
    
    base_dir = "dataset"
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    print(f"Successfully downloaded {len(images)} face images.")
    
    half = len(images) // 2
    
    print(f"Saving {half} 'Real' face images...")
    for i in range(half):
        # Convert from RGB (sklearn) to BGR (OpenCV)
        img_bgr = cv2.cvtColor((images[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(real_dir, f"real_{i}.jpg"), img_bgr)
        
    print(f"Generating {len(images) - half} 'Fake' face images...")
    for i in range(half, len(images)):
        img_bgr = cv2.cvtColor((images[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # --- Simulate a "Deepfake" ---
        # We simulate a bad deepfake by taking the center of the face (where swaps happen)
        # and applying artifacts (blurring and compression artifacts)
        h, w, channels = img_bgr.shape
        center_h, center_w = h // 2, w // 2
        radius_h, radius_w = h // 3, w // 3
        
        # Extract the facial region
        face_region = img_bgr[center_h-radius_h : center_h+radius_h, center_w-radius_w : center_w+radius_w]
        
        if face_region.size > 0:
            # Apply a heavy blur (simulating blending errors in deepfakes)
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
            
            # Apply pixelation/compression artifacts (simulating GAN artifacts)
            small = cv2.resize(blurred_face, (10, 10), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (face_region.shape[1], face_region.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Blend it back into the original image
            img_bgr[center_h-radius_h : center_h+radius_h, center_w-radius_w : center_w+radius_w] = pixelated
            
        cv2.imwrite(os.path.join(fake_dir, f"fake_{i}.jpg"), img_bgr)
        
    print("\n✅ Mini-dataset creation complete!")
    print(f"Dataset saved in: {os.path.abspath(base_dir)}")
    print("You can now run 'python train.py' to train your model!")

if __name__ == "__main__":
    generate_mini_dataset()
