import os
from PIL import Image

def resize_images(input_dir, output_dir, target_size=(640, 640)):
    """
    Resizes all images in the input directory to the target size and saves them to the output directory.

    Args:
        input_dir (str): Path to the directory containing the images to resize.
        output_dir (str): Path to the directory where resized images will be saved.
        target_size (tuple): Target size for resizing (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_resized.save(output_path)
                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Specify the input and output directories
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(base_path, 'training_inputs', 'chessred2k_YOLO', 'train', 'images')
    output_directory = os.path.join(base_path, 'training_inputs', 'chessred2k_YOLO', 'train', 'images_resized')

    # Resize images
    resize_images(input_directory, output_directory)