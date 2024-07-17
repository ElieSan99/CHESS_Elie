import os
from collections import defaultdict

def count_images_in_dataset(dataset_path):
    # Initialize a dictionary to store the counts
    image_counts = defaultdict(int)

    # List the main directories (train, val, test)
    main_dirs = ['train', 'val', 'test']
    
    for main_dir in main_dirs:
        main_dir_path = os.path.join(dataset_path, main_dir)
        
        # List all subdirectories (each corresponding to a character)
        if os.path.exists(main_dir_path):
            char_dirs = [d for d in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, d))]
            
            for char_dir in char_dirs:
                char_dir_path = os.path.join(main_dir_path, char_dir)
                
                # Count the number of files (images) in each character directory
                num_images = len([f for f in os.listdir(char_dir_path) if os.path.isfile(os.path.join(char_dir_path, f))])
                image_counts[char_dir] += num_images

    return image_counts

dataset_path = 'new_dataset'  # Replace with the path to your dataset
image_counts = count_images_in_dataset(dataset_path)

# Print the results
for char, count in image_counts.items():
    print(f"Caractère {char}: {count} images")
