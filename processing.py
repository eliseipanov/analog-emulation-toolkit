import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab, lab2rgb


def _fft_add_grain(channel, cutoff, grain_amount, rng):
    """
    Додає синтетичне зерно до високих частот, НЕ вирізаючи оригінал.
    Низькі частоти (< cutoff) залишаються незмінними.
    Високі частоти (> cutoff) зберігаються, але до них додається шум.
    """
    h, w = channel.shape
    fft = np.fft.fftshift(np.fft.fft2(channel))

    cy, cx = h // 2, w // 2
    radius = min(cy, cx) * cutoff
    y, x = np.ogrid[:h, :w]
    high_mask = (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2

    # Енергія оригінальних високих частот (для масштабування шуму)
    orig_high_energy = np.sqrt(np.mean(np.abs(fft[high_mask]) ** 2)) + 1e-8

    # Генеруємо шум у просторовій області, потім у частотній
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))
    noise_energy = np.sqrt(np.mean(np.abs(noise_fft[high_mask]) ** 2)) + 1e-8

    # Масштабуємо шум так, щоб він був часткою енергії оригіналу
    noise_fft = noise_fft / noise_energy * orig_high_energy * grain_amount
    noise_fft = noise_fft * high_mask

    # ДОДАЄМО шум до оригіналу, не вирізаючи нічого
    combined = fft + noise_fft
    out = np.real(np.fft.ifft2(np.fft.ifftshift(combined)))
    return out


def _grain_luminance_mask(gray):
    return np.clip(1.0 - np.abs(gray - 0.45) * 2.0, 0.0, 1.0)


def _subpixel_row_shift(img, max_shift, rng):
    """Горизонтальний sub-pixel зсув кожного рядка (імітація нерівного руху головки сканера)."""
    h, w, c = img.shape
    shifts = rng.uniform(-max_shift, max_shift, h).astype(np.float32)

    x = np.arange(w, dtype=np.float32)
    out = np.empty_like(img)
    for row in range(h):
        s = shifts[row]
        src_x = x - s
        x0 = np.floor(src_x).astype(np.int32)
        x1 = x0 + 1
        frac = src_x - x0
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        for ch in range(c):
            out[row, :, ch] = img[row, x0, ch] * (1 - frac) + img[row, x1, ch] * frac
    return out


def _lab_chroma_shift(img, strength, rng):
    """Зміна хроматичних каналів a/b у LAB-просторі."""
    if strength <= 0:
        return img
    lab = rgb2lab(img)
    h, w = lab.shape[:2]
    noise_a = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    noise_b = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    lab[:, :, 1] += noise_a * strength
    lab[:, :, 2] += noise_b * strength
    lab[:, :, 1] = np.clip(lab[:, :, 1], -100, 100)
    lab[:, :, 2] = np.clip(lab[:, :, 2], -100, 100)
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def _apply_hot_pixels(img, count, rng):
    h, w, c = img.shape
    if count <= 0:
        return img
    ys = rng.integers(0, h, count)
    xs = rng.integers(0, w, count)
    for y, x in zip(ys, xs):
        img[y, x] = 1.0
    return img


def _apply_dead_pixels(img, count, rng):
    h, w, c = img.shape
    if count <= 0:
        return img
    ys = rng.integers(0, h, count)
    xs = rng.integers(0, w, count)
    for y, x in zip(ys, xs):
        img[y, x] = 0.0
    return img


def _apply_dust_spots(img, count, rng):
    h, w, c = img.shape
    if count <= 0:
        return img
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(count):
        cy = rng.integers(0, h)
        cx = rng.integers(0, w)
        radius = rng.uniform(2.0, 10.0)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        spot = np.clip(1.0 - dist / radius, 0.0, 1.0)
        spot = gaussian_filter(spot, sigma=max(0.5, radius / 2.0))
        intensity = rng.uniform(0.15, 0.35)
        mask = np.maximum(mask, spot * intensity)
    mask = gaussian_filter(mask, sigma=2.0)
    mask = np.clip(mask, 0.0, 1.0)
    for ch in range(c):
        img[:, :, ch] *= (1.0 - mask)
    return img


def _apply_sensor_shading(img, strength, rng):
    h, w, c = img.shape
    if strength <= 0:
        return img
    small_h = max(2, h // 16)
    small_w = max(2, w // 16)
    noise = rng.normal(0.0, 1.0, (small_h, small_w)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=1.0)
    nmin = noise.min()
    nmax = noise.max()
    if nmax > nmin:
        noise = (noise - nmin) / (nmax - nmin)
    else:
        noise = np.zeros_like(noise)
    repeat_y = max(1, h // small_h)
    repeat_x = max(1, w // small_w)
    noise = np.repeat(np.repeat(noise, repeat_y, axis=0), repeat_x, axis=1)
    noise = noise[:h, :w]
    noise = gaussian_filter(noise, sigma=max(1.0, min(h, w) / 64.0))
    factor = 1.0 - strength * noise
    factor = np.clip(factor, 1.0 - strength, 1.0)
    for ch in range(c):
        img[:, :, ch] *= factor
    return img


def apply_film_scan_processing(image, frequency_cutoff, grain_amount, seed,
                                row_shift=0.0, lab_chroma_strength=0.0,
                                hot_pixels=0, dead_pixels=0, dust_spots=0, sensor_shading=0.0):
    rng = np.random.default_rng(seed)
    processed = []

    for img in image:
        img_np = img.cpu().numpy()
        h, w, c = img_np.shape
        gray = np.mean(img_np, axis=2)
        grain_mask = _grain_luminance_mask(gray)

        # 1. FFT: додаємо зерно до високих частот, не вирізаючи оригінал
        out = np.zeros_like(img_np)
        for ch in range(c):
            channel = img_np[:, :, ch]
            processed_ch = _fft_add_grain(channel, frequency_cutoff, grain_amount, rng)
            # Просторовий шум за яскравістю (дуже малий)
            noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32) * 0.005
            processed_ch = processed_ch + noise * grain_mask
            out[:, :, ch] = processed_ch
        out = np.clip(out, 0.0, 1.0)

        # 2. Sub-pixel зсув рядків
        if row_shift > 0:
            out = _subpixel_row_shift(out, row_shift, rng)

        # 3. LAB-обробка хроматичності
        if lab_chroma_strength > 0:
            out = _lab_chroma_shift(out, lab_chroma_strength, rng)

        # 4. Sensor defects
        if hot_pixels > 0:
            out = _apply_hot_pixels(out, hot_pixels, rng)
        if dead_pixels > 0:
            out = _apply_dead_pixels(out, dead_pixels, rng)
        if dust_spots > 0:
            out = _apply_dust_spots(out, dust_spots, rng)
        if sensor_shading > 0:
            out = _apply_sensor_shading(out, sensor_shading, rng)

        processed.append(torch.from_numpy(out).float())

    return torch.stack(processed)