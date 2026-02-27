import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_images(container_path, watermark_path):
    """ Загружает RGB, преобразует в YCrCb. Загружает и бинаризует водяной знак"""
    container_bgr = cv2.imread(container_path)
    container_rgb = cv2.cvtColor(container_bgr, cv2.COLOR_BGR2RGB)

    container_ycbcr = cv2.cvtColor(container_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(container_ycbcr)

    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    _, watermark_bin = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)

    h, w = Cb.shape
    watermark_bin = cv2.resize(watermark_bin, (w, h))
    
    return container_rgb, Y, Cr, Cb, watermark, watermark_bin


def embed_simple_qim(Cb, watermark_bin, delta):
    """ Встраивание ЦВЗ """
    C = Cb.astype(np.int32)
    W = watermark_bin.astype(np.int32)

    # U = C mod δ  (формула 3.13)
    U = C % delta

    # Cw = floor(C/(2δ)) * 2δ + W*δ + U
    Cw = (C // (2 * delta)) * (2 * delta) + W * delta + U
    Cw = np.clip(Cw, 0, 255).astype(np.uint8)
    return Cw


def extract_simple_qim(Cb_watermarked, delta):
    """ Извлечение ЦВЗ """
    Cw_int = Cb_watermarked.astype(np.int32)

    remainder = Cw_int % (2 * delta)
    extracted = (remainder >= delta).astype(np.uint8)

    return extracted


def main():
    delta = 12
    # container_path = "goldhill.tif"
    container_path = "baboon.tif"
    watermark_path = "ornament.tif"

    # Загрузка
    container_rgb, Y, Cr, Cb, watermark, watermark_bin = load_images(
        container_path, watermark_path
    )

    # Встраивание
    Cb_watermarked = embed_simple_qim(Cb, watermark_bin, delta)

    # Сборка изображения
    watermarked_ycbcr = cv2.merge((Y, Cr, Cb_watermarked))
    watermarked_rgb = cv2.cvtColor(watermarked_ycbcr, cv2.COLOR_YCrCb2RGB)

    # Извлечение
    extracted = extract_simple_qim(Cb_watermarked, delta)
    extracted_vis = extracted * 255


    # ВИЗУАЛИЗАЦИЯ
    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.title("Исходное изображение")
    plt.imshow(container_rgb)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Изменённый канал Cb")
    plt.imshow(Cb_watermarked, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Изображение с ЦВЗ")
    plt.imshow(watermarked_rgb)
    plt.axis("off")

    plt.show()

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Оригинальный ЦВЗ")
    plt.imshow(watermark_bin * 255, cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Извлечённый ЦВЗ")
    plt.imshow(extracted_vis, cmap="gray")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()