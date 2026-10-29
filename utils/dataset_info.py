import os


DATASET_PATH = "dataset"


EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def count_images(folder_path):
    """Count image files inside a folder."""
    
    image_extensions = (".jpg", ".jpeg", ".png")
    
    count = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            count += 1
    
    return count


def analyze_dataset():
    """Display the number of images in each emotion class."""
    
    print("\nTRAINING DATASET")
    print("-" * 30)
    
    train_total = 0
    
    for emotion in EMOTIONS:
        folder = os.path.join(DATASET_PATH, "train", emotion)
        
        count = count_images(folder)
        
        print(f"{emotion:10} : {count}")
        
        train_total += count
    
    print("-" * 30)
    print(f"Total      : {train_total}")
    
    
    print("\nTEST DATASET")
    print("-" * 30)
    
    test_total = 0
    
    for emotion in EMOTIONS:
        folder = os.path.join(DATASET_PATH, "test", emotion)
        
        count = count_images(folder)
        
        print(f"{emotion:10} : {count}")
        
        test_total += count
    
    print("-" * 30)
    print(f"Total      : {test_total}")


if __name__ == "__main__":
    analyze_dataset()