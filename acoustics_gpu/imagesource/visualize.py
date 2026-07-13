import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import torch
from typing import List, Tuple
from .core import (
    get_virtual_sources_torch,
    calculate_paths_and_energies_torch,
    fit_energy_curves_torch,
    FREQ_LABELS,
    FREQ_COUNT,
)
from ..common.utils import device

def draw_room_and_sources_torch(room_width: float,
                                room_height: float,
                                source: List[float],
                                receiver: List[float],
                                max_order: int,
                                reflection_coeffs_freq: dict[str, List[float]]):
    print(f"\n{'=' * 60}")
    print("Начало вычислений на GPU...")
    start_time = time.time()

    # 1. Вычисляем виртуальные источники на GPU
    virtual_sources_dict = get_virtual_sources_torch(
        source, room_width, room_height, max_order, reflection_coeffs_freq
    )

    n_sources = virtual_sources_dict['positions'].shape[0]
    print(f"Сгенерировано виртуальных источников: {n_sources}")

    # 2. Вычисляем пути и энергии на GPU
    results = calculate_paths_and_energies_torch(
        virtual_sources_dict, receiver, room_width, room_height
    )

    distances = results['distances']
    energies_matrix = results['energies']  # [n_sources, 6]
    positions = results['positions']
    orders = results['orders']
    intersections_list = results['intersections']

    gpu_time = time.time() - start_time
    print(f"Вычисления на GPU завершены за {gpu_time:.3f} секунд")
    print(f"Среднее время на источник: {gpu_time / n_sources * 1000:.3f} мс")

    # 3. Создаем график помещения
    fig, ax = plt.subplots(figsize=(14, 12))

    # Рисуем помещение
    room_patch = patches.Rectangle((0, 0), room_width, room_height, linewidth=2,
                                   edgecolor='orange', facecolor='none', linestyle='-')
    ax.add_patch(room_patch)

    # Источник и приемник
    ax.plot(source[0], source[1], 'ro', markersize=12, label='Источник', zorder=10)
    ax.plot(receiver[0], receiver[1], 'go', markersize=12, label='Приёмник', zorder=10)
    ax.plot([source[0], receiver[0]], [source[1], receiver[1]],
            color='purple', linestyle='-', linewidth=3, alpha=0.8, label='Прямой путь', zorder=5)

    # 4. Рисуем пути отражений
    colors = plt.cm.rainbow(np.linspace(0, 1, n_sources))

    for i in range(n_sources):
        pos = positions[i]
        order = orders[i]

        # Строим путь с учетом пересечений
        path_points = [pos]
        if intersections_list[i]:
            for inter in intersections_list[i]:
                path_points.append([inter[0], inter[1]])
        path_points.append(receiver)

        # Рисуем путь
        plot_reflection_path(ax, path_points, color=colors[i % len(colors)])

        # Виртуальный источник
        ax.plot(pos[0], pos[1], 'ro', markersize=6, alpha=0.7, zorder=5)

        # Виртуальное помещение
        offset_x = (pos[0] // room_width) * room_width
        offset_y = (pos[1] // room_height) * room_height
        virtual_room = patches.Rectangle(
            (offset_x, offset_y), room_width, room_height,
            linewidth=1, edgecolor='black', facecolor='none',
            linestyle='--', alpha=0.3
        )
        ax.add_patch(virtual_room)

    # Настройки графика
    ax.set_xlim(-room_width, 2 * room_width)
    ax.set_ylim(-room_height, 2 * room_height)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'Моделирование акустики помещения (GPU ускорение)\n'
                 f'Максимальный порядок: {max_order}, Источников: {n_sources}\n'
                 f'Время вычислений на GPU: {gpu_time:.3f} с',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 5. Выводим статистику для каждого луча
    print(f"\n{'=' * 60}")
    print("СТАТИСТИКА ЛУЧЕЙ:")
    print(f"{'Порядок':<10} {'Длина (м)':<12} {'Время (мс)':<12} {'Энергия (средняя)':<20}")
    print("-" * 60)

    for i in range(min(10, n_sources)):  # Показываем первые 10 лучей
        avg_energy = np.mean(energies_matrix[i])
        time_ms = results['times'][i] * 1000
        print(f"{orders[i]:<10} {distances[i]:<12.2f} {time_ms:<12.2f} {avg_energy:<20.6f}")

    if n_sources > 10:
        print(f"... и ещё {n_sources - 10} лучей")

    # 6. Аппроксимация зависимостей энергии от расстояния для каждой частоты
    print(f"\n{'=' * 60}")
    print("АППРОКСИМАЦИЯ ЗАВИСИМОСТИ ЭНЕРГИИ ОТ РАССТОЯНИЯ:")

    # Используем PyTorch для быстрой аппроксимации
    C_values = fit_energy_curves_torch(distances, energies_matrix)

    # Создаем отдельные графики для каждой частоты
    for freq_idx in range(FREQ_COUNT):
        energies_freq = energies_matrix[:, freq_idx]

        if len(energies_freq) > 1 and np.any(energies_freq > 0):
            plt.figure(figsize=(12, 8))

            # Точки данных
            plt.scatter(distances, energies_freq,
                        c=energies_freq, cmap='viridis', alpha=0.7,
                        s=50, edgecolors='black', linewidth=0.5,
                        label=f'Данные ({FREQ_LABELS[freq_idx]})')

            # Аппроксимирующая кривая
            C = C_values[freq_idx]
            r_smooth = np.linspace(np.min(distances), np.max(distances), 500)
            e_smooth = C / (r_smooth ** 2)

            plt.plot(r_smooth, e_smooth, 'r-', linewidth=3,
                     label=f'Аппроксимация: E(r) = {C:.4f} / r²')

            # Настройки графика
            plt.title(f'Зависимость энергии от расстояния ({FREQ_LABELS[freq_idx]})\n'
                      f'Коэффициент C = {C:.4f}', fontsize=14, fontweight='bold')
            plt.xlabel('Расстояние (м)', fontsize=12)
            plt.ylabel('Энергия', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=11)
            plt.colorbar(label='Уровень энергии')

            plt.tight_layout()
            plt.show()

            print(f"{FREQ_LABELS[freq_idx]:<10} C = {C:.6f}")

    # 7. Сводный график для всех частот
    plt.figure(figsize=(14, 10))

    for freq_idx in range(FREQ_COUNT):
        energies_freq = energies_matrix[:, freq_idx]
        if np.any(energies_freq > 0):
            C = C_values[freq_idx]
            r_smooth = np.linspace(np.min(distances), np.max(distances), 200)
            e_smooth = C / (r_smooth ** 2)

            plt.plot(r_smooth, e_smooth, linewidth=2.5,
                     label=f'{FREQ_LABELS[freq_idx]} (C={C:.3f})')

    plt.title('Сводная аппроксимация для всех частотных диапазонов\n'
              f'E(r) = C / r²', fontsize=16, fontweight='bold')
    plt.xlabel('Расстояние (м)', fontsize=14)
    plt.ylabel('Энергия', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Частотные диапазоны', title_fontsize=12, fontsize=11)
    plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации
    plt.tight_layout()
    plt.show()

    # 8. Анализ производительности GPU
    if torch.cuda.is_available():
        print(f"\n{'=' * 60}")
        print("ИНФОРМАЦИЯ О GPU:")
        print(f"Устройство: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU выделено: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"Память GPU кэшировано: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

    return {
        'n_sources': n_sources,
        'computation_time': gpu_time,
        'C_values': C_values,
        'distances': distances,
        'energies': energies_matrix
    }

def plot_reflection_path(ax, points: List[Tuple[float, float]], color: str = 'y'):
    xs, ys = zip(*points)
    ax.plot(xs, ys, linestyle='-', linewidth=1.5, color=color, alpha=0.7)
    for p in points[1:-1]:
        ax.plot(p[0], p[1], 'bo', markersize=6, alpha=0.7)
