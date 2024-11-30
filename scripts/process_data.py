from PIL import Image
import os

def crop_images(input_folder, output_folder, coordinates):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(root, file)
                relative_output_path = os.path.relpath(file_path, input_folder)
                output_path = os.path.join(output_folder, relative_output_path)

                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    img = Image.open(file_path)
                    cropped_img = img.crop(coordinates)
                    cropped_img.save(output_path)
                    print(f'Cropped and saved: {output_path}')
                except Exception as e:
                    print(f'Error processing {file}: {e}')

def resize_image(folder_path, output_folder, target_size=(224, 224)):
    try:
        for filename in os.listdir(folder_path):
            # Check if the file is an image
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                # Open the image file
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)

                # Resize the image
                resized_img = img.resize(target_size)

                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f'Cropped and saved: {output_path}')
    except Exception as e:
        print(f"Error occurred: {e}")


def convert_images_to_grayscale(folder_path, output_folder):
    try:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a PNG image
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                # Open the PNG image
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)

                # Convert the image to grayscale
                grayscale_img = img.convert('L')

                # Save the grayscale image to the output folder
                output_path = os.path.join(output_folder, filename)
                grayscale_img.save(output_path)
                print(f'Cropped and saved: {output_path}')
    except Exception as e:
        print(f"Error occurred: {e}")

