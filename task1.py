# Реализовать встраивание ЦВЗ в указанную в варианте задания битовую плоскость определённого цветового канала пустого контейнера.
# Визуализировать результат встраивания: как итоговое изображение, так и отдельно изменённый цветовой канал.


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



CONTAINER_PATH = 'goldhill.tif'
WATERMARK1_PATH = 'ornament.tif'
WATERMARK2_PATH = 'mickey.tif'
OUTPUT_PATH = 'output/'





def load_image(path, grayscale=False):
    if grayscale: img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else: img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"not File: {path}")
    print(f"loaded {path}: size {img.shape}, type {img.dtype}")
    return img


def prepare_watermark(watermark_path, target_shape):
    wm = load_image(watermark_path, grayscale=True)
    watermark_original = wm.copy()

    h, w = target_shape[:2]# Изменяем размер под контейнер
    wm_resized = cv2.resize(wm, (w, h), interpolation=cv2.INTER_NEAREST)

    _, wm_bin = cv2.threshold(wm_resized, 128, 1, cv2.THRESH_BINARY) # делаем строго бинарным
    wm_bin = wm_bin.astype(np.uint8)

    print(f"Watermark успешно подготовлен, размер {wm_bin.shape}")
    return wm_bin, watermark_original


def embed_into_bit_plane(container, watermark_bin, channel_idx, plane):
    stego = container.copy()

    original_channel = container[:, :, channel_idx].copy()
    channel = stego[:, :, channel_idx].copy()

    # Сдвигаем биты ЦВЗ на нужную позицию
    watermark_shifted = watermark_bin << plane

    # C_p^W = C_p XOR W  - ТОЧНО ПО МЕТОДИЧКЕ!
    channel_modified = channel ^ watermark_shifted

    stego[:, :, channel_idx] = channel_modified
    difference = cv2.absdiff(original_channel, channel_modified)

    return stego, channel_modified, original_channel, difference


def visualize_embedding(container, stego, channel_name, plane,original_ch, modified_ch, watermark_bin,
                        watermark_original, difference, filename):


    fig = plt.figure(figsize=(16, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(container, cv2.COLOR_BGR2RGB))
    plt.title('Исходный контейнер', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB))
    plt.title(f'Стего ({channel_name}-{plane})', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(original_ch, cmap='gray')
    plt.title(f'Исходный {channel_name} канал', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow(modified_ch, cmap='gray')
    plt.title(f'Модиф. {channel_name} канал', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 4, 5)
    plt.imshow(watermark_original, cmap='gray')
    plt.title('Оригинальный ЦВЗ', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 4, 6)
    plt.imshow(watermark_bin * 255, cmap='gray')
    plt.title('Бинарный ЦВЗ (0/1)', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 4, 7)
    plt.imshow(difference * 30, cmap='gray')
    plt.title(f'Разница (x30)\nмакс={difference.max()}', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 4, 8)
    plt.hist(original_ch.ravel(), bins=50, alpha=0.7, label='Исходный', color='blue')
    plt.hist(modified_ch.ravel(), bins=50, alpha=0.7, label='Модиф.', color='red')
    plt.title('Гистограмма канала', fontsize=10)
    plt.legend(fontsize=8)


    save_path = os.path.join(OUTPUT_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()



def main():
    try:
        container = load_image(CONTAINER_PATH)

        wm_green_bin, wm_green_orig = prepare_watermark(WATERMARK1_PATH, container.shape)
        wm_blue_bin, wm_blue_orig = prepare_watermark(WATERMARK2_PATH, container.shape)

        print(" Встраивание в Green-2...")
        stego_after_green, green_modified, green_original, green_diff = embed_into_bit_plane(
            container, wm_green_bin, 1,2  # Индекс зеленого канала в BGR (0-Blue, 1-Green, 2-Red)
        )

        print("\nПРОВЕРКА Green-2")
        print("Количество изменённых пикселей:", np.sum(green_diff != 0))
        visualize_embedding(
            container, stego_after_green,
            'Green', 2,
            green_original, green_modified,
            wm_green_bin, wm_green_orig, green_diff,
            'embedding_green2.png'
        )
        print(" Встраивание в Blue-1...")
        stego_final, blue_modified, blue_original, blue_diff = embed_into_bit_plane(
            stego_after_green, wm_blue_bin, 0, 1
        )
        print("Количество изменённых пикселей:", np.sum(blue_diff != 0))
        visualize_embedding(
            stego_after_green, stego_final,
            'Blue', 1,
            blue_original, blue_modified,
            wm_blue_bin, wm_blue_orig, blue_diff,
            'embedding_blue1.png'
        )

        final_path = os.path.join(OUTPUT_PATH, 'stego_task1_14var.png')
        cv2.imwrite(final_path, stego_final)
        print(f"стего-изображение сохранено: {final_path}")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return


if __name__ == "__main__":
    main()