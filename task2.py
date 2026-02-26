# Извлечение информации для 14 варианта

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

GREEN_CHANNEL = 1
BLUE_CHANNEL = 0
GREEN_PLANE = 2
BLUE_PLANE = 1

CONTAINER_PATH = 'baboon.tif'
STEGO_PATH = 'output/stego_task1_14var.png'
OUTPUT_PATH = 'output/'


def extract_xor(original_container, stego_image, channel_idx, plane):
    original_channel = original_container[:, :, channel_idx]
    stego_channel = stego_image[:, :, channel_idx]

    # XOR каналов
    xor_result = stego_channel ^ original_channel
    # Извлекаем нужную битовую плоскость
    extracted = (xor_result >> plane) & 1
    
    return extracted


def visualize_extraction(extracted, channel_name, plane, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(extracted * 255, cmap='gray')
    plt.title(f'Извлечено из {channel_name}-{plane}')
    plt.axis('off')

    save_path = os.path.join(OUTPUT_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Сохранено: {save_path}")


def main():
    original_container = cv2.imread(CONTAINER_PATH)
    stego_image = cv2.imread(STEGO_PATH)

    if original_container is None or stego_image is None:
        print("Ошибка загрузки изображений")
        return

    extracted_green = extract_xor(original_container, stego_image, GREEN_CHANNEL, GREEN_PLANE)
    extracted_blue = extract_xor( original_container, stego_image, BLUE_CHANNEL, BLUE_PLANE)
    visualize_extraction(extracted_green, "Green", GREEN_PLANE, "extracted_green2.png")
    visualize_extraction(extracted_blue, "Blue", BLUE_PLANE, "extracted_blue1.png")


if __name__ == "__main__":
    main()