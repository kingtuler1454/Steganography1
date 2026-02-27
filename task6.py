"""
На основе выполненных заданий 1-2 из основного списка реализо-вать стеганографическое встраивание
 в НЗБ полутонового контей-нера текстовой информации с последующим её извлечением (СВИ-2).
 Способ преобразования текста в бинарный вектор не принципиален и оставляется на усмотрение студента.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

CONTAINER_PATH = 'goldhill.tif'  # Полутоновое изображение
OUTPUT_PATH = 'output/'
TEXT_FILE = 'secret_message.txt'  # Файл с текстом для встраивания
EXTRACTED_TEXT_FILE = 'extracted_message.txt'  # Извлеченный текст


def load_grayscale_image(path):
    """ Загрузка полутонового изображения """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Файл не найден: {path}")
    print(f"Загружено изображение: {img.shape}, тип {img.dtype}")
    return img


def text_to_bits(text):
    """
    Преобразование текста в битовый вектор
    Каждый символ -> 8 бит (ASCII)
    Добавляем специальный маркер конца сообщения
    """
    text_with_end = text + "\0\0\0\0"

    bits = []
    for char in text_with_end:
        char_code = ord(char)
        # Получаем 8 бит символа (от старшего к младшему)
        for i in range(7, -1, -1):
            bits.append((char_code >> i) & 1)

    return np.array(bits, dtype=np.uint8)


def bits_to_text(bits):
    """ Преобразование битового вектора обратно в текст """
    if len(bits) % 8 != 0:
        bits = bits[:len(bits) - len(bits) % 8]

    chars = []

    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte |= bits[i + j] << (7 - j)

        chars.append(byte)

        # Проверяем последние 4 байта
        if len(chars) >= 4 and chars[-4:] == [0, 0, 0, 0]:
            return ''.join(chr(c) for c in chars[:-4])

    return ''.join(chr(c) for c in chars)


def embed_bits_into_lsb(container, bits, start_position=0):
    """
    Встраивание битов в НЗБ полутонового изображения
    Использует метод XOR из задания 1: C' = C XOR bits
    """
    stego = container.copy().astype(np.uint16)
    flat_container = stego.flatten()

    # Проверяем, хватит ли места
    if start_position + len(bits) > len(flat_container):
        raise ValueError(f"Недостаточно места в контейнере. Нужно {len(bits)} бит, "
                         f"доступно {len(flat_container) - start_position}")

    for i, bit in enumerate(bits):
        idx = start_position + i
        flat_container[idx] ^= bit

    stego = flat_container.reshape(container.shape).astype(np.uint8)
    return stego


def extract_bits_from_lsb(stego, original_container, num_bits):
    """
    Извлечение битов из НЗБ полутонового изображения
    Использует метод XOR из задания 2: bits = stego XOR original
    """
    flat_stego = stego.flatten()
    flat_original = original_container.flatten()

    xor_result = flat_stego ^ flat_original
    extracted_bits = xor_result & 1

    return extracted_bits[:num_bits]


def read_text_file(filename):
    """ Чтение текста из файла """
    with open(filename, 'r', encoding='ascii') as f:
        return f.read()


def visualize_lsb_embedding(original, stego, text, extracted_text, filename):
    """Визуализация результатов встраивания текста"""

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))


    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Оригинал (полутоновый)')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(stego, cmap='gray')
    axes[0, 1].set_title('Стего с текстом')
    axes[0, 1].axis('off')

    diff = cv2.absdiff(original, stego)
    axes[0, 2].imshow(diff * 255, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f'Разница (x30)\nизменено: {np.sum(diff > 0)} пикс.')
    axes[0, 2].axis('off')

    lsb_original = original & 1
    axes[0, 3].imshow(lsb_original * 255, cmap='gray')
    axes[0, 3].set_title('НЗБ оригинала')
    axes[0, 3].axis('off')
    lsb_stego = stego & 1
    axes[1, 0].imshow(lsb_stego * 255, cmap='gray')
    axes[1, 0].set_title('НЗБ стего')
    axes[1, 0].axis('off')

    # 6. Гистограмма
    axes[1, 1].hist(original.ravel(), bins=50, alpha=0.7, label='Оригинал', color='blue')
    axes[1, 1].hist(stego.ravel(), bins=50, alpha=0.7, label='Стего', color='red')
    axes[1, 1].set_title('Гистограмма')
    axes[1, 1].legend(fontsize=8)

    # 7. Текст (оригинал)
    axes[1, 2].text(0.1, 0.9, f'Оригинальный текст:\n{text[:100]}...',
                    fontsize=9, wrap=True, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

    # 8. Текст (извлеченный)
    axes[1, 3].text(0.1, 0.9, f'Извлеченный текст:\n{extracted_text[:100]}...',
                    fontsize=9, wrap=True, transform=axes[1, 3].transAxes)
    axes[1, 3].axis('off')

    plt.suptitle('СВИ-2: Встраивание текста в НЗБ полутонового изображения', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Визуализация сохранена: {save_path}")


def calculate_capacity(image_shape):
    """Расчет емкости контейнера"""
    total_pixels = image_shape[0] * image_shape[1]
    total_bytes = total_pixels // 8
    total_chars = total_bytes  # 1 байт на символ

    return {
        'pixels': total_pixels,
        'bits': total_pixels,
        'bytes': total_bytes,
        'chars': total_chars
    }


def main():
    print("=" * 60)
    print("СВИ-2: Встраивание текста в НЗБ полутонового изображения")
    print("=" * 60)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    try:
        print("\nЗагрузка контейнера...")
        container = load_grayscale_image(CONTAINER_PATH)

        capacity = calculate_capacity(container.shape)
        print(f"\n Емкость контейнера:")
        print(f"   Пикселей: {capacity['pixels']}")
        print(f"   Битов: {capacity['bits']}")
        print(f"   Байт: {capacity['bytes']}")
        print(f"   Символов (ASCII): {capacity['chars']}")

        print("\nПодготовка текстового сообщения...")
        text = read_text_file(os.path.join(OUTPUT_PATH, TEXT_FILE))
        print(f"   Текст длиной {len(text)} символов")

        bits = text_to_bits(text)
        print(f"   Битов для встраивания: {len(bits)}")

        if len(bits) > capacity['bits']:
            print(f"Ошибка: текст слишком длинный!")
            print(f"   Нужно бит: {len(bits)}, доступно: {capacity['bits']}")
            return

        print("\nВстраивание текста...")
        stego = embed_bits_into_lsb(container, bits)

        stego_path = os.path.join(OUTPUT_PATH, 'stego_text_lsb.png')
        cv2.imwrite(stego_path, stego)
        print(f"Стего сохранено: {stego_path}")

        print("\nИзвлечение текста...")
        extracted_bits = extract_bits_from_lsb(stego, container, len(bits))
        extracted_text = bits_to_text(extracted_bits)

        extracted_path = os.path.join(OUTPUT_PATH, EXTRACTED_TEXT_FILE)
        with open(extracted_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"Извлеченный текст сохранен: {extracted_path}")

        print("\nСоздание визуализации...")
        visualize_lsb_embedding(container, stego, text, extracted_text, 'text_lsb_visualization.png')

        print("\nСтатистика:")
        diff = cv2.absdiff(container, stego)
        print(
            f"   Изменено пикселей: {np.sum(diff > 0)} из {container.size} ({np.sum(diff > 0) / container.size * 100:.2f}%)")
        print(f"   Среднее изменение: {diff.mean():.2f}")
        print(f"   Макс. изменение: {diff.max()}")

        print("\nДоп. задание 2 успешно выполнено!")

    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()