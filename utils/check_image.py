from PIL import Image
import os


IMAGE_PATH = "dataset/train/angry"


def find_first_image(folder_path):
    """Find the first image inside the given folder."""

    image_extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            return os.path.join(folder_path, filename)

    return None


def inspect_image():
    """Display information about one dataset image."""

    image_path = find_first_image(IMAGE_PATH)

    if image_path is None:
        print("No image found.")
        return

    image = Image.open(image_path)

    print("\nIMAGE INFORMATION")
    print("-" * 30)

    print("File       :", image_path)
    print("Format     :", image.format)
    print("Size       :", image.size)
    print("Mode       :", image.mode)


if __name__ == "__main__":
    inspect_image()