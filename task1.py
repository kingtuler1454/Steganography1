# Реализовать встраивание ЦВЗ в указанную в варианте задания битовую плоскость
# определённого цветового канала пустого контейнера.
# Визуализировать результат встраивания: как итоговое изображение, так и отдельно изменённый цветовой канал.

"""
task1.py - Лабораторная работа 1, задание 1
14 вариант: встраивание в Green-2 и Blue-1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Константы для 14 варианта
GREEN_CHANNEL = 1  # Индекс зеленого канала в BGR (0-Blue, 1-Green, 2-Red)
BLUE_CHANNEL = 0  # Индекс синего канала
GREEN_PLANE = 2  # 2-й бит для зеленого
BLUE_PLANE = 1  # 1-й бит для синего

# Пути к файлам
CONTAINER_PATH = 'baboon.tif'
WATERMARK1_PATH = 'ornament.tif'  # Для зеленого канала
WATERMARK2_PATH = 'mickey.tif'  # Для синего канала
OUTPUT_PATH = 'output/'


def create_output_dir():
    """Создает папку для результатов, если её нет"""
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"✅ Создана папка {OUTPUT_PATH}")


def load_image(path, grayscale=False):
    """
    Загружает изображение
    grayscale=True - загружает как черно-белое
    grayscale=False - загружает как цветное (BGR)
    """
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"❌ Файл не найден: {path}")

    print(f"✅ Загружен {path}: размер {img.shape}, тип {img.dtype}")
    return img


def prepare_watermark(watermark_path, target_shape):
    """
    Подготавливает ЦВЗ для встраивания:
    1. Изменяет размер под размер контейнера
    2. Превращает в бинарное изображение (0 и 1)

    Возвращает:
    - watermark_bin: бинарное изображение для встраивания (0 и 1)
    - watermark_original: оригинальное изображение для визуализации
    """
    # Загружаем ЦВЗ как черно-белое
    wm = load_image(watermark_path, grayscale=True)
    watermark_original = wm.copy()

    # Изменяем размер под контейнер
    h, w = target_shape[:2]
    wm_resized = cv2.resize(wm, (w, h), interpolation=cv2.INTER_NEAREST)

    # Бинаризуем: всё что больше 128 -> 1, иначе -> 0
    _, wm_bin = cv2.threshold(wm_resized, 128, 1, cv2.THRESH_BINARY)
    wm_bin = wm_bin.astype(np.uint8)

    print(f"✅ ЦВЗ подготовлен: размер {wm_bin.shape}, значения 0 и {np.unique(wm_bin)}")
    return wm_bin, watermark_original


def embed_into_bit_plane(container, watermark_bin, channel_idx, plane):
    """
    Встраивает бинарный ЦВЗ в указанную битовую плоскость указанного канала

    Параметры:
    - container: исходное изображение (BGR)
    - watermark_bin: бинарное изображение (0 и 1) того же размера, что и container
    - channel_idx: индекс канала (0-Blue, 1-Green, 2-Red)
    - plane: номер битовой плоскости (0-7, где 0 - LSB)

    Возвращает:
    - stego: изображение с встроенным ЦВЗ
    - modified_channel: измененный канал
    - original_channel: исходный канал
    - difference: разница между исходным и измененным каналом
    """
    # Создаем копию контейнера
    stego = container.copy()

    # Получаем целевой канал
    original_channel = container[:, :, channel_idx].copy()
    channel = stego[:, :, channel_idx].copy()

    # Создаем маску для обнуления целевого бита
    # Пример: для plane=2 (3-й бит): 11111011 (в двоичной)
    # mask = ~(1 << plane) & 0xFF  # & 0xFF для 8-битного числа
    # Обнуляем целевой бит
    # channel_cleared = channel & mask
    
    # Сдвигаем биты ЦВЗ на нужную позицию
    watermark_shifted = watermark_bin << plane
    # Побитовое сложение (XOR)
    channel_modified = channel ^ watermark_shifted

    # Комбинируем: очищенный канал + биты ЦВЗ
    # channel_modified = channel_cleared | watermark_shifted

    # Возвращаем измененный канал в изображение
    stego[:, :, channel_idx] = channel_modified

    # Вычисляем разницу
    difference = cv2.absdiff(original_channel, channel_modified)

    return stego, channel_modified, original_channel, difference


def visualize_embedding(container, stego, channel_name, plane,
                        original_ch, modified_ch, watermark_bin,
                        watermark_original, difference, filename):
    """
    Визуализирует результаты встраивания
    """
    # Создаем фигуру с подграфиками
    fig = plt.figure(figsize=(16, 8))

    # 1. Исходный контейнер
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(container, cv2.COLOR_BGR2RGB))
    plt.title('Исходный контейнер', fontsize=10)
    plt.axis('off')

    # 2. Стего-изображение
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB))
    plt.title(f'Стего ({channel_name}-{plane})', fontsize=10)
    plt.axis('off')

    # 3. Исходный канал
    plt.subplot(2, 4, 3)
    plt.imshow(original_ch, cmap='gray')
    plt.title(f'Исходный {channel_name} канал', fontsize=10)
    plt.axis('off')

    # 4. Модифицированный канал
    plt.subplot(2, 4, 4)
    plt.imshow(modified_ch, cmap='gray')
    plt.title(f'Модиф. {channel_name} канал', fontsize=10)
    plt.axis('off')

    # 5. Оригинальный ЦВЗ
    plt.subplot(2, 4, 5)
    plt.imshow(watermark_original, cmap='gray')
    plt.title('Оригинальный ЦВЗ', fontsize=10)
    plt.axis('off')

    # 6. Бинарный ЦВЗ
    plt.subplot(2, 4, 6)
    plt.imshow(watermark_bin * 255, cmap='gray')
    plt.title('Бинарный ЦВЗ (0/1)', fontsize=10)
    plt.axis('off')

    # 7. Разница (увеличена для наглядности)
    # Разница обязана повторять бинарный watermark
    plt.subplot(2, 4, 7)
    #  imshow по умолчанию всегда растягивает масштаб, 
    # поэтому чтобы увидеть  чб изображение можно :
    # plt.imshow(difference, cmap='gray')
    plt.imshow(difference * 30, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Разница (x30)\nмакс={difference.max()}', fontsize=10)
    plt.axis('off')

    # 8. Гистограмма изменений
    plt.subplot(2, 4, 8)
    plt.hist(original_ch.ravel(), bins=50, alpha=0.7, label='Исходный', color='blue')
    plt.hist(modified_ch.ravel(), bins=50, alpha=0.7, label='Модиф.', color='red')
    plt.title('Гистограмма канала', fontsize=10)
    plt.legend(fontsize=8)

    plt.suptitle(f'Встраивание в {channel_name}-{plane} (14 вариант)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Сохраняем
    save_path = os.path.join(OUTPUT_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Визуализация сохранена: {save_path}")


# def verify_extraction(stego_image, channel_idx, plane, original_watermark_bin, channel_name):
    # """
    # Проверяет корректность извлечения
    # """
    # # Извлекаем биты
    # original_channel = original_container[:, :, channel_idx]
    # stego_channel = stego_image[:, :, channel_idx]

    # xor_result = stego_channel ^ original_channel
    # extracted = (xor_result >> plane) & 1

    # # Сравниваем
    # if np.array_equal(extracted_bits, original_watermark_bin):
    #     print(f"✅ {channel_name}: извлечение успешно - все биты совпадают")
    #     return True
    # else:
    #     mismatch = np.sum(extracted_bits != original_watermark_bin)
    #     percent = (mismatch / original_watermark_bin.size) * 100
    #     print(f"❌ {channel_name}: ошибка - {mismatch} несовпадений ({percent:.2f}%)")
    #     return False


def main():
    """Основная функция"""
    print("=" * 60)
    print("ЗАДАНИЕ 1: Встраивание в битовые плоскости")
    print("14 вариант: Green-2 и Blue-1")
    print("=" * 60)

    # Создаем папку для результатов
    create_output_dir()

    try:
        # 1. Загружаем контейнер
        print("\n📥 Загрузка файлов...")
        container = load_image(CONTAINER_PATH)

        # 2. Загружаем и подготавливаем ЦВЗ
        wm_green_bin, wm_green_orig = prepare_watermark(WATERMARK1_PATH, container.shape)
        wm_blue_bin, wm_blue_orig = prepare_watermark(WATERMARK2_PATH, container.shape)

        # 3. Встраиваем в зеленый канал (Green-2)
        print("\n🟢 Встраивание в Green-2...")
        stego_after_green, green_modified, green_original, green_diff = embed_into_bit_plane(
            container, wm_green_bin, GREEN_CHANNEL, GREEN_PLANE
        )

        print("\nПРОВЕРКА Green-2")
        print("Количество изменённых пикселей:", np.sum(green_diff != 0))

        # Визуализируем результат для зеленого
        visualize_embedding(
            container, stego_after_green,
            'Green', GREEN_PLANE,
            green_original, green_modified,
            wm_green_bin, wm_green_orig, green_diff,
            'embedding_green2.png'
        )

        # 4. Встраиваем в синий канал (Blue-1)
        print("\n🔵 Встраивание в Blue-1...")
        stego_final, blue_modified, blue_original, blue_diff = embed_into_bit_plane(
            stego_after_green, wm_blue_bin, BLUE_CHANNEL, BLUE_PLANE
        )

        # Визуализируем результат для синего
        visualize_embedding(
            stego_after_green, stego_final,
            'Blue', BLUE_PLANE,
            blue_original, blue_modified,
            wm_blue_bin, wm_blue_orig, blue_diff,
            'embedding_blue1.png'
        )

        # 5. Сохраняем финальное изображение
        final_path = os.path.join(OUTPUT_PATH, 'stego_task1_14var.png')
        cv2.imwrite(final_path, stego_final)
        print(f"\n💾 Финальное стего-изображение сохранено: {final_path}")

        # 6. Проверяем извлечение
        # print("\n🔍 Проверка извлечения...")
        # verify_extraction(stego_final, GREEN_CHANNEL, GREEN_PLANE, wm_green_bin, "Green-2")
        # verify_extraction(stego_final, BLUE_CHANNEL, BLUE_PLANE, wm_blue_bin, "Blue-1")

        # 7. Финальная визуализация
        print("\n📊 Создание финальной визуализации...")
        plt.figure(figsize=(15, 8))

        # Исходный контейнер
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(container, cv2.COLOR_BGR2RGB))
        plt.title('Исходный контейнер')
        plt.axis('off')

        # Финальное стего
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(stego_final, cv2.COLOR_BGR2RGB))
        plt.title('Финальное стего\n(Green-2 + Blue-1)')
        plt.axis('off')

        # Общая разница
        total_diff = cv2.absdiff(container, stego_final)
        total_diff_gray = cv2.cvtColor(total_diff, cv2.COLOR_BGR2GRAY)
        plt.subplot(2, 3, 3)
        plt.imshow(total_diff_gray * 30, cmap='gray')
        plt.title('Общая разница (x30)')
        plt.axis('off')

        # Модифицированный зеленый
        plt.subplot(2, 3, 4)
        plt.imshow(stego_final[:, :, GREEN_CHANNEL], cmap='gray')
        plt.title('Модиф. Green канал')
        plt.axis('off')

        # Модифицированный синий
        plt.subplot(2, 3, 5)
        plt.imshow(stego_final[:, :, BLUE_CHANNEL], cmap='gray')
        plt.title('Модиф. Blue канал')
        plt.axis('off')

        # Информация о варианте
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, '14 вариант', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.6, 'Green-2: ornament.tif', fontsize=12)
        plt.text(0.1, 0.4, 'Blue-1: mickey.tif', fontsize=12)
        plt.text(0.1, 0.2, f'Размер: {container.shape[1]}x{container.shape[0]}', fontsize=10)
        plt.axis('off')

        plt.suptitle('Задание 1 - Результат для 14 варианта', fontsize=16)
        plt.tight_layout()

        final_viz_path = os.path.join(OUTPUT_PATH, 'final_visualization.png')
        plt.savefig(final_viz_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Финальная визуализация сохранена: {final_viz_path}")
        print("\n🎉 Задание 1 успешно выполнено!")
        print(f"Все результаты сохранены в папке: {OUTPUT_PATH}")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return


if __name__ == "__main__":
    main()