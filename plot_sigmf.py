#!/usr/bin/env python3
# plot_sigmf_waterfall.py

import os
import json
import mmap # Не використовується прямо в цьому варіанті, але може бути корисним для дуже великих файлів
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from tqdm import tqdm # Не використовується прямо, але імпортовано в оригінальному файлі

# ────────────────────────────────────────────────────────────── helpers ──
def load_meta(meta_path):
    with open(meta_path, "r", encoding='utf-8') as f: # Додано encoding
        meta = json.load(f)
    g = meta["global"]
    fs   = g["core:sample_rate"]
    fc   = g.get("core:frequency", 0)
    dt   = g["core:datatype"].lower()
    # core:num_samples не завжди присутнє в 'global', може бути специфічним для запису (capture)
    # Для загального випадку, краще не покладатися на нього тут для визначення розміру файлу.
    # Розмір буде визначено з самого файлу даних.
    nsmp = g.get("core:num_samples") 
    return fs, fc, dt, nsmp

def mmap_sigmf(data_path, datatype, num_samples_meta=None): # num_samples_meta - з метаданих, якщо є
    """Повертає np.memmap або np.ndarray як complex64 незалежно від вихідного типу даних."""
    
    # Визначення розміру файлу та кількості семплів на основі розміру файлу
    file_size_bytes = os.path.getsize(data_path)
    bytes_per_sample = 0
    calculated_num_samples = 0

    if datatype in ("cf32_le", "cf32"): # Додано cf32 без _le/_be
        dt = np.dtype("<c8" if datatype.endswith("le") else ">c8")
        bytes_per_sample = dt.itemsize
        if bytes_per_sample > 0:
            calculated_num_samples = file_size_bytes // bytes_per_sample
        mm = np.memmap(data_path, dtype=dt, mode="r", shape=(calculated_num_samples,))
        return mm.astype(np.complex64, copy=False)

    if datatype in ("cf64_le", "cf64"): # Додано cf64 без _le/_be
        dt = np.dtype("<c16" if datatype.endswith("le") else ">c16")
        bytes_per_sample = dt.itemsize
        if bytes_per_sample > 0:
            calculated_num_samples = file_size_bytes // bytes_per_sample
        mm = np.memmap(data_path, dtype=dt, mode="r", shape=(calculated_num_samples,))
        return mm.astype(np.complex64, copy=False)

    if datatype in ("ci8", "ci16_le", "ci16_be", "ci16"): # Додано ci16 без _le/_be
        norm = 0.0
        dt_raw = None
        if datatype == "ci8":
            dt_raw = np.int8
            norm = 128.0
            bytes_per_sample = 2 # I + Q
        elif datatype.startswith("ci16"):
            dt_raw = np.dtype("<i2" if datatype.endswith("le") or datatype == "ci16" else ">i2") # ci16 за замовчуванням le
            norm = 32768.0
            bytes_per_sample = 4 # I + Q
        
        if bytes_per_sample > 0:
            calculated_num_samples = file_size_bytes // (bytes_per_sample // 2) # //2 бо dt_raw для одного компонента (I або Q)

        # Читання всього файлу, якщо num_samples_meta не вказано або для перевірки
        raw = np.memmap(data_path, dtype=dt_raw, mode="r")

        # Якщо num_samples_meta вказано і воно менше, ніж розраховано з розміру файлу,
        # можна було б обмежити, але безпечніше використовувати розрахункову кількість.
        # calculated_num_samples тут це кількість I або Q компонентів, не пар IQ.
        # Нам потрібно 2 * кількість_IQ_семплів
        if calculated_num_samples % 2 != 0: # Має бути парна кількість I, Q компонентів
            raw = raw[:calculated_num_samples -1]
        
        num_iq_samples = raw.size // 2
            
        iq  = raw.reshape(num_iq_samples, 2).astype(np.float32) / norm
        sig = iq[:, 0] + 1j * iq[:, 1]
        return sig.astype(np.complex64, copy=False) # Повертаємо complex64

    raise ValueError(f"Непідтриманий core:datatype = {datatype}")


def compute_waterfall(sig, fs, nfft=4096, overlap_ratio=0.9, max_rows_display=4096):
    nfft = min(nfft, len(sig)) # Страховка, якщо сигнал коротший за nfft
    if nfft == 0: # Якщо сигнал порожній
        # Повертаємо порожні масиви відповідної форми, щоб уникнути помилок далі
        print("Попередження: Сигнал порожній або занадто короткий для FFT.")
        return np.array([]), np.array([]), np.array([[]])


    noverlap = int(nfft * overlap_ratio)
    freqs, times, S = spectrogram(
        sig,
        fs=fs,
        window="hann",
        nperseg=nfft,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False, # Повертає двосторонній спектр (-fs/2 до fs/2)
        axis=0, # Обробка вздовж першої осі (якщо sig багатовимірний, хоча тут він 1D)
        scaling="density",
        mode="magnitude",
    )
    
    # Зменшення кількості часових відрізків (стовпців S), якщо їх забагато для відображення
    # S має форму (кількість_частот, кількість_часових_відрізків)
    if S.shape[1] > max_rows_display: # max_rows_display тут фактично обмежує кількість часових відрізків
        step = int(np.ceil(S.shape[1] / max_rows_display))
        S    = S[:, ::step]
        times= times[::step]
        
    S_db = 20 * np.log10(S + 1e-12) # Додаємо мале число, щоб уникнути log10(0)
    
    # Сортування частот, щоб 0 Гц був у центрі (fftshift еквівалент)
    idx  = np.argsort(freqs) 
    freqs_sorted = freqs[idx]
    S_db_sorted = S_db[idx, :] # Сортуємо рядки S_db відповідно до частот
    
    return freqs_sorted, times, S_db_sorted

def plot_waterfall(freqs, times, S_db, fc, out_png=None):
    # freqs - відсортовані частоти (від негативних до позитивних, 0 Гц в центрі)
    # times - часові мітки
    # S_db - дані спектрограми (потужність в dB), форма: (кількість_частот, кількість_часових_відрізків)
    # fc - центральна частота
    
    if S_db.size == 0 or freqs.size == 0 or times.size == 0:
        print("Помилка: Немає даних для відображення водоспаду.")
        return

    plt.figure(figsize=(12, 7), dpi=120) # Трохи змінив розмір для кращого вигляду
    
    # Координати для осей графіку
    # Вісь X - Частота в МГц
    x_axis_coords_MHz = (freqs + fc) / 1e6
    # Вісь Y - Час в секундах
    y_axis_coords_sec = times

    # Extent для imshow: [x_min, x_max, y_min, y_max]
    # y-вісь часу буде інвертована (час_початку вгорі)
    current_extent = [
        x_axis_coords_MHz[0],    # Мін. частота для осі X
        x_axis_coords_MHz[-1],   # Макс. частота для осі X
        y_axis_coords_sec[-1],   # Час внизу осі Y (останній часовий відрізок)
        y_axis_coords_sec[0],    # Час вгорі осі Y (перший часовий відрізок)
    ]

    # Для відповідності осям (X=Частота, Y=Час), дані для imshow мають бути (час, частота)
    # S_db має форму (частоти, час), тому його потрібно транспонувати S_db.T
    data_for_imshow = S_db.T # Тепер форма (кількість_часових_відрізків, кількість_частот)

    plt.imshow(
        data_for_imshow,
        extent=current_extent,
        aspect="auto",
        cmap="viridis", # Популярна і інформативна колірна схема
        vmin=np.percentile(S_db, 5), 
        vmax=np.percentile(S_db, 95),
        origin='upper' # 'upper' означає, що data_for_imshow[0,0] в лівому верхньому куті.
                       # Оскільки y_axis_coords_sec[0] (початковий час) - це верхня межа extent, це правильно.
    )
    plt.xlabel("Частота, МГц")
    plt.ylabel("Час, с")
    plt.title(f"Waterfall SigMF (Fc: {fc/1e6:.3f} МГц)") # Додав Fc в заголовок
    plt.colorbar(label="Потужність, дБ") # Змінив dB на дБ

    # --- Додано: Пошук сигналів за рівнем та обведення ---
    # 1. Визначення порогу
    # Можна експериментувати з цим значенням.
    #threshold_value = np.percentile(S_db, 90) # Шукаємо 10% найяскравіших сигналів
    threshold_value = -145
    # Або фіксоване значення, наприклад:
    # v_min_plot = np.percentile(S_db, 5)
    # v_max_plot = np.percentile(S_db, 95)
    # threshold_value = v_min_plot + (v_max_plot - v_min_plot) * 0.75 # 75% від динамічного діапазону на графіку
    
    print(f"Використовується поріг для контурів: {threshold_value:.2f} дБ")

    # 2. Малювання контурів
    # plt.contour(X, Y, Z, levels, ...)
    # X - координати для стовпців Z (у нас це частоти - x_axis_coords_MHz)
    # Y - координати для рядків Z (у нас це час - y_axis_coords_sec)
    # Z - дані, на яких шукаємо контури (мають бути data_for_imshow або S_db.T)
    
    if data_for_imshow.shape[0] > 1 and data_for_imshow.shape[1] > 1: # contour потребує хоча б 2x2
        plt.contour(
            x_axis_coords_MHz,   # Координати для осі X (Частоти)
            y_axis_coords_sec,   # Координати для осі Y (Час)
            data_for_imshow,     # Дані (транспоновані S_db: час x частоти)
            levels=[threshold_value], # Рівень(ні) для малювання контурів
            colors='red',        # Колір ліній
            linewidths=0.75,     # Товщина ліній
            alpha=0.8            # Прозорість ліній
        )
    else:
        print("Попередження: Недостатньо даних для малювання контурів (потрібно мінімум 2x2).")
    # --- Кінець доданого коду ---
    
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"✅ PNG збережено → {out_png}")
    plt.show()

# ────────────────────────────────────────────────────────────── main ────
def main(meta_path_str):
    meta_path = str(meta_path_str) # Переконуємося, що це рядок для сумісності
    data_path = meta_path.replace(".sigmf-meta", ".sigmf-data")
    
    if not os.path.exists(meta_path):
        print(f"Помилка: Файл метаданих не знайдено: {meta_path}")
        return
    if not os.path.exists(data_path):
        print(f"Помилка: Файл даних не знайдено: {data_path}")
        return

    fs, fc, datatype, nsamples_meta = load_meta(meta_path)
    print(
        f"Тип даних: {datatype}, Fs = {fs/1e6:.2f} Msps, Fc = {fc/1e6:.3f} МГц",
        f"({nsamples_meta or 'не вказано в мета'} семплів в мета)",
    )

    # Завантаження сигналу
    # nsamples_meta тут може бути None, mmap_sigmf обробить це, читаючи весь файл
    sig = mmap_sigmf(data_path, datatype, nsamples_meta) 
    
    if sig.size == 0:
        print("Помилка: Не вдалося завантажити дані сигналу або сигнал порожній.")
        return

    print(f"Завантажено {sig.size} IQ семплів.")

    # Параметри для спектрограми - можна винести в аргументи командного рядка
    nfft_val = 4096
    overlap_ratio_val = 0.9 # 90% перекриття
    
    print(f"Обчислення водоспаду з nfft={nfft_val}, перекриття={overlap_ratio_val*100}%...")
    freqs, times, S_db = compute_waterfall(sig, fs, nfft=nfft_val, overlap_ratio=overlap_ratio_val)
    
    # Визначення імені вихідного файлу
    base_name = os.path.basename(meta_path)
    name_without_ext = os.path.splitext(base_name)[0]
    # Переконаємося, що вихідний файл зберігається в поточній директорії або вказаній
    output_directory = "." # Поточна директорія
    png_filename = os.path.join(output_directory, name_without_ext + "_waterfall.png")
    
    plot_waterfall(freqs, times, S_db, fc, out_png=png_filename)

if __name__ == "__main__":
    import argparse
    # sys тут не потрібен, якщо argparse використовується
    parser = argparse.ArgumentParser(description="Генерує та відображає водоспад з файлу SigMF.")
    parser.add_argument("meta", help="Шлях до файлу *.sigmf-meta")
    # Можна додати більше аргументів, наприклад, --nfft, --overlap сюди
    
    args = parser.parse_args()
    main(args.meta)