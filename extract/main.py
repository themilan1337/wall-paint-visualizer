import fitz  # PyMuPDF
import re
import json
import os

PDF_PATH = "main.pdf"
OUTPUT_JSON = "colors.json"

# Regex для кодов цветов, например: N002, S001, H003
CODE_REGEX = re.compile(r"^[A-Z]\d{3}$")

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b).upper()

def rgb_to_lab(r, g, b):
    # Ручная конвертация RGB -> LAB без использования тяжелых библиотек (skimage/numpy)
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    r = r * 100.0
    g = g * 100.0
    b = b * 100.0

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x = x / 95.047
    y = y / 100.000
    z = z / 108.883

    x = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    y = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    z = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)

    L = (116 * y) - 16
    A = 500 * (x - y)
    B = 200 * (y - z)

    return [round(L, 2), round(A, 2), round(B, 2)]

def extract_colors():
    results = {}
    doc = fitz.open(PDF_PATH)
    total = len(doc)
    
    found = 0
    for page_index in range(total):
        page = doc[page_index]
        
        # 1. Собрать все цветовые плашки (векторные заливки) из PDF
        swatches = []
        for p in page.get_drawings():
            if p["fill"]: # Если у вектора есть заливка
                rect = p["rect"]
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                
                # Фильтр по размеру (плашки примерно 198x85)
                # Берем с запасом: ширина > 100, высота > 50
                if w > 100 and h > 50:
                    swatches.append({
                        "rect": rect,
                        "fill": p["fill"]
                    })
                    
        # Если плашек нет (например, страница-оглавление), пропускаем
        if not swatches:
            continue
            
        # 2. Собрать все тексты-коды (например, N002)
        codes = []
        text_dict = page.get_text("dict")
        if "blocks" in text_dict:
            for block in text_dict["blocks"]:
                if block["type"] == 0:  # text
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if CODE_REGEX.fullmatch(text):
                                codes.append({
                                    "code": text,
                                    "bbox": span["bbox"]
                                })
                                
        # 3. Сопоставить коды и плашки
        # В PDF плашка находится чуть ВЫШЕ текста
        for code_info in codes:
            ty0 = code_info["bbox"][1] # Верхняя граница текста
            best_swatch = None
            min_dist = 999
            
            for swatch in swatches:
                sy1 = swatch["rect"][3] # Нижняя граница плашки
                dist = ty0 - sy1
                
                # Ищем плашку, которая находится прямо над текстом (дистанция от 0 до 20)
                if 0 <= dist < 20:
                    if dist < min_dist:
                        min_dist = dist
                        best_swatch = swatch
                        
            if best_swatch:
                # Извлекаем оригинальный векторный цвет
                fill = best_swatch["fill"]
                r = int(round(fill[0] * 255))
                g = int(round(fill[1] * 255))
                b = int(round(fill[2] * 255))
                
                code = code_info["code"]
                hex_val = rgb_to_hex(r, g, b)
                lab_val = rgb_to_lab(r, g, b)
                
                results[code] = {
                    "code": code,
                    "hex": hex_val,
                    "rgb": [r, g, b],
                    "lab": lab_val,
                    "page": page_index + 1
                }
                found += 1
                
        # Выводим прогресс раз в 20 страниц, чтобы не спамить
        if (page_index + 1) % 20 == 0 or (page_index + 1) == total:
            print(f"Обработано {page_index + 1}/{total} страниц. Найдено цветов: {found}", flush=True)

    return results

if __name__ == "__main__":
    print(f"🚀 Старт: Извлекаем 100% точные векторные цвета из {PDF_PATH}...", flush=True)
    if not os.path.isfile(PDF_PATH):
        print(f"Ошибка: файл {PDF_PATH} не найден.", flush=True)
        exit(1)
        
    data = extract_colors()

    # Сортируем по алфавиту
    sorted_data = {k: data[k] for k in sorted(data.keys())}

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Готово! Успешно извлечено {len(sorted_data)} цветов -> {OUTPUT_JSON}", flush=True)
