"""
Простой тест для проверки установки библиотеки acoustics_gpu
"""

print("=" * 50)
print("ТЕСТИРОВАНИЕ БИБЛИОТЕКИ acoustics_gpu")
print("=" * 50)

# 1. Проверка импорта
try:
   n 
    print("✅ Импорт acoustics_gpu успешен")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    exit(1)

# 2. Проверка версии
try:
    print(f"   Версия: {acoustics_gpu.__version__}")
except AttributeError:
    print("   ⚠️ __version__ не определён")

# 3. Проверка импорта компонентов
print("\n--- Проверка компонентов ---")

components = [
    'MATERIAL_LIBRARY',
    'FREQUENCY_BANDS',
    'N_FREQUENCIES',
    'FREQ_LABELS',
    'SPEED_OF_SOUND',
    'EnergyMapOptimizer',
    'trace_ray_spectral',
]

for comp in components:
    try:
        getattr(acoustics_gpu, comp)
        print(f"  ✅ {comp}")
    except AttributeError:
        print(f"  ❌ {comp} не найден")

# 4. Проверка Ray Tracing
print("\n--- Ray Tracing ---")
try:
    from acoustics_gpu.raytracing import EnergyMapOptimizer
    print("  ✅ EnergyMapOptimizer импортирован")
except ImportError as e:
    print(f"  ❌ Ошибка: {e}")

# 5. Проверка Image Source
print("\n--- Image Source ---")
try:
    from acoustics_gpu.imagesource import get_virtual_sources_torch
    print("  ✅ get_virtual_sources_torch импортирован")
except ImportError as e:
    print(f"  ❌ Ошибка: {e}")

print("\n" + "=" * 50)
print("ТЕСТ ЗАВЕРШЁН")