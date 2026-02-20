#!/usr/bin/env python3
"""Тест базы данных цветов"""

from color_database import color_db

# Загружаем базу
print("Loading color database...")
color_db.load()

print(f"\nTotal colors: {color_db.get_count()}")

# Тестируем поиск
print("\n=== Test 1: Search for 'F00' ===")
results = color_db.search('F00', limit=5)
for color in results:
    print(f"  {color['code']}: {color['hex']} RGB{color['rgb']}")

print("\n=== Test 2: Search for 'K05' ===")
results = color_db.search('K05', limit=5)
for color in results:
    print(f"  {color['code']}: {color['hex']} RGB{color['rgb']}")

print("\n=== Test 3: Get specific color 'F001' ===")
color = color_db.get_by_code('F001')
if color:
    print(f"  {color['code']}: {color['hex']} RGB{color['rgb']}")
else:
    print("  Not found")

print("\n=== Test 4: First 5 colors (empty search) ===")
results = color_db.search('', limit=5)
for color in results:
    print(f"  {color['code']}: {color['hex']} RGB{color['rgb']}")

print("\nAll tests passed!")
