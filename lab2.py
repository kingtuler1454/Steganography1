import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os

np.random.seed(42)


# ==================== БАЗОВЫЕ ФУНКЦИИ ====================

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Изображение {path} не найдено")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(img.shape) == 3 else img.astype(np.float32)


def dct2(x): return dct(dct(x.T, norm='ortho').T, norm='ortho')


def idct2(x): return idct(idct(x.T, norm='ortho').T, norm='ortho')


def generate_watermark(length, alpha=1.0):
    return np.random.normal(0, 1, length) * alpha


def calculate_psnr(orig, wm):
    mse = np.mean((orig - wm) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def detect_watermark(orig, extracted):
    denom = np.sqrt(np.sum(orig ** 2)) * np.sqrt(np.sum(extracted ** 2))
    return 0 if denom == 0 else np.sum(orig * extracted) / denom


# ==================== ЗОНЫ ДКП ====================

def get_dct_zones(shape):
    h, w = shape
    zones = np.zeros(shape, dtype=int)
    total = h * w
    l_size, m_size = total // 16, 3 * total // 16

    count = 0
    for s in range(h + w - 1):
        if s % 2 == 0:  # четные диагонали (снизу вверх)
            i, j = min(s, h - 1), s - min(s, h - 1)
            while i >= 0 and j < w:
                zones[i, j] = 0 if count < l_size else (1 if count < l_size + m_size else 2)
                count += 1;
                i -= 1;
                j += 1
        else:  # нечетные диагонали (сверху вниз)
            j, i = min(s, w - 1), s - min(s, w - 1)
            while j >= 0 and i < h:
                zones[i, j] = 0 if count < l_size else (1 if count < l_size + m_size else 2)
                count += 1;
                i += 1;
                j -= 1
    return zones


# ==================== ВСТРАИВАНИЕ/ИЗВЛЕЧЕНИЕ ====================

def get_embed_indices(zones, coeff_ratio=0.5):
    m_idx = np.where(zones == 1)
    sorted_order = np.argsort(m_idx[0] + m_idx[1])
    m_sorted = (m_idx[0][sorted_order], m_idx[1][sorted_order])
    n = int(len(m_sorted[0]) * coeff_ratio)
    return (m_sorted[0][:n], m_sorted[1][:n]), n


def embed_additive(dct_coeffs, omega, indices):
    wm_dct = dct_coeffs.copy()
    wm_dct[indices] += omega
    return wm_dct


def extract_watermark(orig_dct, wm_dct, indices):
    return wm_dct[indices] - orig_dct[indices]


# ==================== ОСНОВНОЙ ПРОЦЕСС ====================

def process_alpha(alpha, img, dct_orig, zones, n_coeffs, coeff_ratio, temp_path):
    indices, _ = get_embed_indices(zones, coeff_ratio)
    omega = generate_watermark(n_coeffs, alpha)

    wm_dct = embed_additive(dct_orig, omega[:n_coeffs], indices)
    wm_img = np.clip(idct2(wm_dct), 0, 255).astype(np.uint8)
    cv2.imwrite(temp_path, wm_img)

    loaded = read_image(temp_path)
    extracted = extract_watermark(dct_orig, dct2(loaded), indices)
    os.remove(temp_path)

    return {
        'alpha': alpha,
        'rho': detect_watermark(omega[:n_coeffs], extracted),
        'psnr': calculate_psnr(img, wm_img),
        'wm_img': wm_img,
        'omega': omega[:n_coeffs],
        'extracted': extracted
    }


def find_optimal_alpha(alphas, img, dct_orig, zones, n_coeffs, coeff_ratio, threshold=0.9):
    results = []
    print(f"\n{'=' * 50}\nПОДБОР α (порог ρ > {threshold})\n{'=' * 50}")

    for a in alphas:
        res = process_alpha(a, img, dct_orig, zones, n_coeffs, coeff_ratio, "temp.tif")
        results.append(res)
        if a % 1 == 0 or res['rho'] > threshold:
            print(f"α={a:4.1f} | ρ={res['rho']:.4f} | PSNR={res['psnr']:.2f}")

    valid = [r for r in results if r['rho'] > threshold]
    if valid:
        optimal = max(valid, key=lambda x: x['psnr'])
        print(f"\n✅ Оптимальное α={optimal['alpha']:.1f} (PSNR={optimal['psnr']:.2f}, ρ={optimal['rho']:.4f})")
    else:
        optimal = max(results, key=lambda x: x['rho'])
        print(f"\n⚠️ Порог не достигнут, макс ρ={optimal['rho']:.4f} при α={optimal['alpha']:.1f}")

    return results, optimal


# ==================== ИССЛЕДОВАНИЯ ====================

def false_detection_study(img, dct_orig, zones, n_coeffs, alpha, coeff_ratio, n_trials=100):
    print(f"\n{'=' * 50}\nИССЛЕДОВАНИЕ: Ложное обнаружение\n{'=' * 50}")

    indices, _ = get_embed_indices(zones, coeff_ratio)
    omega_true = generate_watermark(n_coeffs, alpha)

    wm_dct = embed_additive(dct_orig, omega_true, indices)
    wm_img = np.clip(idct2(wm_dct), 0, 255).astype(np.uint8)
    cv2.imwrite("temp_false.tif", wm_img)

    loaded = read_image("temp_false.tif")
    extracted = extract_watermark(dct_orig, dct2(loaded), indices)
    os.remove("temp_false.tif")

    rho_true = detect_watermark(omega_true, extracted)
    print(f"Корреляция с настоящим ЦВЗ: ρ = {rho_true:.6f}")

    false_rhos = []
    for i in range(n_trials):
        rho = detect_watermark(np.random.normal(0, 1, n_coeffs), extracted)
        false_rhos.append(rho)
        if (i + 1) % 20 == 0:
            print(f"  Выполнено {i + 1}/{n_trials}")

    false_rhos = np.array(false_rhos)

    # Статистика
    print(f"\nСтатистика по {n_trials} случайным ЦВЗ:")
    print(f"  Среднее: {np.mean(false_rhos):.4f} | Std: {np.std(false_rhos):.4f}")
    print(f"  Мин: {np.min(false_rhos):.4f} | Макс: {np.max(false_rhos):.4f}")

    for thr in [0.5, 0.7, 0.9]:
        fp = np.sum(false_rhos > thr)
        print(f"  Ложных срабатываний (ρ>{thr}): {fp}/{n_trials} ({fp / n_trials * 100:.1f}%)")

    rank = np.sum(false_rhos > rho_true) + 1
    print(f"\nНастоящий ЦВЗ занимает {rank}-е место из {n_trials + 1}")

    # Визуализация
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.hist(false_rhos, bins=30, alpha=0.7, density=True)
    plt.axvline(rho_true, color='r', linewidth=2, label=f'Настоящий (ρ={rho_true:.3f})')
    plt.axvline(np.mean(false_rhos), color='g', ls='--', label=f'Среднее={np.mean(false_rhos):.3f}')
    plt.xlabel('ρ');
    plt.ylabel('Плотность');
    plt.title('Распределение корреляций');
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(132)
    plt.plot(false_rhos, 'o', markersize=3, alpha=0.5)
    plt.axhline(rho_true, color='r', linewidth=2)
    for thr, c in zip([0.5, 0.7, 0.9], ['orange', 'g', 'purple']):
        plt.axhline(thr, color=c, ls='--', alpha=0.5, label=f'Порог {thr}')
    plt.xlabel('Эксперимент');
    plt.ylabel('ρ');
    plt.title('Корреляции случайных ЦВЗ');
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(133)
    plt.boxplot(false_rhos, patch_artist=True)
    plt.axhline(rho_true, color='r', linewidth=2, label='Настоящий')
    plt.xticks([1], ['Случайные ЦВЗ']);
    plt.ylabel('ρ');
    plt.title('Статистика');
    plt.legend();
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('false_detection.png', dpi=150)
    plt.show()

    return false_rhos


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def visualize(initial, all_results):
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(initial['wm_img'], cmap='gray')
    plt.title(f'С ЦВЗ (α={initial["alpha"]:.1f})\nPSNR={initial["psnr"]:.2f}')
    plt.axis('off')

    plt.subplot(232)
    diff = np.abs(initial['wm_img'].astype(float) - read_image("bridge.tif").astype(float)) * 10
    plt.imshow(diff, cmap='hot')
    plt.title('Разность (x10)')
    plt.axis('off')

    plt.subplot(233)
    alphas = [r['alpha'] for r in all_results]
    plt.plot(alphas, [r['rho'] for r in all_results], 'b-', label='ρ(α)')
    plt.axhline(0.9, color='r', ls='--', label='Порог')
    plt.xlabel('α');
    plt.ylabel('ρ');
    plt.title('ρ от α');
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(234)
    plt.plot(alphas, [r['psnr'] for r in all_results], 'g-')
    plt.xlabel('α');
    plt.ylabel('PSNR');
    plt.title('PSNR от α');
    plt.grid(alpha=0.3)

    plt.subplot(235)
    plt.hist(initial['omega'], bins=50, alpha=0.5, label='Исходный')
    plt.hist(initial['extracted'], bins=50, alpha=0.5, label='Извлеченный')
    plt.xlabel('Значение');
    plt.ylabel('Частота');
    plt.title('Гистограммы ЦВЗ')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('lab2_results.png', dpi=150)
    plt.show()


# ==================== MAIN ====================

def main():
    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №2: Встраивание ЦВЗ (Вариант 5)")
    print("ДКП, зона M, аддитивный метод, доля 1/2")
    print("=" * 60)

    # Инициализация
    img = read_image("bridge.tif")
    dct_orig = dct2(img)
    zones = get_dct_zones(img.shape)
    _, n_coeffs = get_embed_indices(zones, 0.5)

    print(f"\nИзображение: {img.shape}, зоны: L={np.sum(zones == 0)}, M={np.sum(zones == 1)}, H={np.sum(zones == 2)}")
    print(f"Длина ЦВЗ: {n_coeffs}")

    # Пункты 1-6
    print("\n" + "-" * 40)
    print("ПУНКТЫ 1-6: Начальное встраивание (α=5.0)")
    initial = process_alpha(5.0, img, dct_orig, zones, n_coeffs, 0.5, "temp.tif")
    print(f"PSNR = {initial['psnr']:.2f}, ρ = {initial['rho']:.6f}")

    # Пункт 7
    all_results, optimal = find_optimal_alpha(
        np.arange(0.5, 20.0, 0.5), img, dct_orig, zones, n_coeffs, 0.5, 0.9
    )

    # Визуализация
    visualize(initial, all_results)

    # Пункт 8
    false_detection_study(img, dct_orig, zones, n_coeffs, optimal['alpha'], 0.5, 100)


if __name__ == "__main__":
    main()