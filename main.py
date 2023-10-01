import tensorflow as tf
import numpy as np
import os
import cv2
import argparse

from PIL import Image


def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

def load_and_resize_image(image, target_size=(224, 224)):

    image = cv2.resize(image, target_size)

    return image

def zoom(image_path):

    image = cv2.imread(image_path)
    new_width, new_height = 1400, 1000
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    x1 = center_x - new_width // 2
    x2 = center_x + new_width // 2
    y1 = center_y - new_height // 2
    y2 = center_y + new_height // 2
    cropped_zoomed_image = image[y1:y2, x1:x2]

    return cropped_zoomed_image

def working(image_path, model):
    
    cropped_zoomed_image = zoom(image_path)
    
    resized_image = load_and_resize_image(cropped_zoomed_image, target_size=(224, 224))

    img_array = tf.expand_dims(resized_image, 0)

    predictions = model.predict(img_array)

    output_path = image_rotation(image_path, (int(predictions[0][0])) * -1, new_name = False)

    return output_path


def image_rotation(input_path, angle, new_name = True):

    image_name = input_path.split('/')[-1]
    root = input_path.split(image_name)[0]

    image = Image.open(input_path)

    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    if new_name:
        output_path = root + str(angle) + '_angle|' + image_name
    else:
        output_path = root + image_name

    rotated_image.save(output_path)

    rotated_image.close()

    return output_path


def interface(img_path):

    output_loc = working(img_path, load_model())
    output_loc = working(output_loc, load_model())
    output_loc = working(output_loc, load_model())
    output_loc = working(output_loc, load_model())
    output_loc = working(output_loc, load_model())


def main(directory):

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                image_path = os.path.join(root, file)

                try:
                    interface(image_path)
                    print(f"Зображення оброблене  - {image_path}")

                except Exception:
                    print(f"Помилка при обробці зображення  - {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Скрипт тестового виводу')
    parser.add_argument('--input', type=str, help='Папка з зображеннями')
    args = parser.parse_args()

    main(args.input)