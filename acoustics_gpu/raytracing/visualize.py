import matplotlib.pyplot as plt
import numpy as np
import torch
from ..common.materials import FREQUENCY_BANDS, N_FREQUENCIES, SPEED_OF_SOUND
from ..common.utils import device

def plot_spectral_energy_maps(energy_maps, x_coords, y_coords, inside_mask, wall_vertices, microphones, source_position,
                              frequencies=FREQUENCY_BANDS):
    """
    Визуализация энергетических карт для разных частот
    """
    n_freqs = len(frequencies)
    n_cols = 3
    n_rows = (n_freqs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Переносим данные на CPU для визуализации
    x_coords_np = x_coords.cpu().numpy()
    y_coords_np = y_coords.cpu().numpy()
    inside_mask_np = inside_mask.cpu().numpy()

    for i, freq_idx in enumerate(range(n_freqs)):
        if i >= len(axes):
            break

        ax = axes[i]

        # Получаем карту для текущей частоты
        energy_map = energy_maps[freq_idx].cpu().numpy()

        # Логарифмическое преобразование для лучшей визуализации
        energy_display = np.log10(energy_map * 100 + 1e-10)
        energy_display[~inside_mask_np] = np.nan

        # Отображаем
        im = ax.imshow(energy_display,
                       extent=[x_coords_np[0], x_coords_np[-1], y_coords_np[0], y_coords_np[-1]],
                       origin='lower', cmap='hot', aspect='auto')

        # Рисуем контур помещения
        wall_x, wall_y = wall_vertices
        ax.plot(np.append(wall_x, wall_x[0]),
                np.append(wall_y, wall_y[0]), 'w-', linewidth=2)

        # Рисуем источник
        ax.scatter(source_position[0], source_position[1], c='red', s=100, marker='*',
                   edgecolors='white', linewidth=2, zorder=5)

        # Рисуем микрофоны
        for mic in microphones:
            mic_x, mic_y = mic['position']
            ax.scatter(mic_x, mic_y, c='blue', s=50, marker='s',
                       edgecolors='white', linewidth=2, zorder=5)

        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_title(f'{frequencies[freq_idx]} Гц')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label='log(Энергия)')

    # Скрываем лишние подграфики
    for i in range(n_freqs, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Энергетические карты для разных частот', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes


def plot_frequency_response(microphones, frequencies=FREQUENCY_BANDS):

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mic in enumerate(microphones):
        if 'frequency_response' in mic:
            freq_response = mic['frequency_response']
            # Нормализуем для отображения в дБ
            if np.max(freq_response) > 0:
                freq_response_db = 10 * np.log10(freq_response / np.max(freq_response) + 1e-10)
            else:
                freq_response_db = 10 * np.log10(freq_response + 1e-10)
            ax.semilogx(frequencies, freq_response_db,
                        'o-', linewidth=2, label=f'Микрофон {i + 1}')

    ax.set_xlabel('Частота (Гц)')
    ax.set_ylabel('Уровень (дБ)')
    ax.set_title('Частотные характеристики в точках микрофонов')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_xlim([min(frequencies), max(frequencies)])

    return fig, ax

def visualize_energy_map_gpu_optimized(energy_maps, x_coords, y_coords, inside_mask, wall_vertices, microphones,
                                       source_position, freq_idx=0, amplification_factor=100.0):
    """
    Оптимизированная визуализация энергетической карты для выбранной частоты
    """
    try:
        # Выбираем карту для конкретной частоты
        if len(energy_maps.shape) == 3:
            energy_matrix = energy_maps[freq_idx]
        else:
            energy_matrix = energy_maps

        energy_processed = torch.log10(energy_matrix * amplification_factor + 1e-10) - 1.8

        energy_inside = energy_processed[inside_mask]

        if len(energy_inside) == 0:
            print("Предупреждение: Нет данных для визуализации")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax1.text(0.5, 0.5, 'Нет данных для визуализации', transform=ax1.transAxes, ha='center', va='center')
            ax2.text(0.5, 0.5, 'Нет данных для гистограммы', transform=ax2.transAxes, ha='center', va='center')
            return fig, (ax1, ax2), {}

        # Используем torch функции для статистики
        total_energy = torch.sum(energy_inside).item()
        mean_energy = torch.mean(energy_inside).item()
        max_energy = torch.max(energy_inside).item()

        threshold_tensor = torch.tensor(1e-8, device=device)
        non_zero_mask = energy_inside > torch.log10(threshold_tensor + 1e-10)
        non_zero_count = torch.sum(non_zero_mask).item()
        total_cells = len(energy_inside)

        # ТОЛЬКО ЗДЕСЬ ПЕРЕНОСИМ НА CPU ДЛЯ MATPLOTLIB
        energy_display = energy_processed.cpu().numpy()
        x_coords_display = x_coords.cpu().numpy()
        y_coords_display = y_coords.cpu().numpy()
        inside_mask_display = inside_mask.cpu().numpy()

        # Создаем графики
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # График 1: 2D энергетическая карта
        energy_for_display = energy_display.copy()
        energy_for_display[~inside_mask_display] = np.nan

        if np.any(np.isfinite(energy_for_display)):
            valid_data = energy_for_display[inside_mask_display & np.isfinite(energy_for_display)]

            if len(valid_data) > 0:
                vmax = np.nanpercentile(valid_data, 95)
                vmin = np.nanmin(valid_data)

                if vmax == vmin:
                    vmax = vmin + 1.0

                im = ax1.imshow(energy_for_display,
                                extent=[x_coords_display[0], x_coords_display[-1],
                                        y_coords_display[0], y_coords_display[-1]],
                                origin='lower', cmap='hot', aspect='auto',
                                vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax1, label='Логарифм энергии')

        # Рисуем контур помещения
        wall_x, wall_y = wall_vertices
        ax1.plot(np.append(wall_x, wall_x[0]),
                 np.append(wall_y, wall_y[0]), 'w-', linewidth=3, label='Помещение')

        # Рисуем источник и микрофоны
        ax1.scatter(source_position[0], source_position[1], c='red', s=150, marker='*',
                    label='Источник', edgecolors='white', linewidth=2, zorder=5)

        for i, mic in enumerate(microphones):
            mic_x, mic_y = mic['position']
            ax1.scatter(mic_x, mic_y, c='blue', s=80, marker='s',
                        label=f'Mic{i + 1}' if i == 0 else "",
                        edgecolors='white', linewidth=2, zorder=5)

        ax1.set_xlabel('X координата (м)')
        ax1.set_ylabel('Y координата (м)')
        ax1.set_title(f'Энергетическая карта (частота {FREQUENCY_BANDS[freq_idx]} Гц, GPU: {device})',
                      fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Гистограмма распределения энергии
        if non_zero_count > 0:
            non_zero_energy = energy_inside[non_zero_mask].cpu().numpy()

            if len(non_zero_energy) > 1:
                ax2.hist(non_zero_energy, bins=min(50, len(non_zero_energy)), alpha=0.7,
                         color='orange', edgecolor='black', linewidth=0.5)

                ax2.axvline(mean_energy, color='red', linestyle='--', linewidth=2,
                            label=f'Среднее: {mean_energy:.2f}')

                ax2.set_xlabel('Логарифм энергии в ячейке')
                ax2.set_ylabel('Количество ячеек')
                ax2.set_title('Распределение энергии по ячейкам', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()

            stats_text = f'Всего ячеек: {total_cells:,}\n' \
                         f'Не нулевых: {non_zero_count:,}\n' \
                         f'Максимум: {max_energy:.2f}\n' \
                         f'Усиление: ×{amplification_factor}'

            ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                     ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        else:
            ax2.text(0.5, 0.5, 'Нет данных для гистограммы',
                     transform=ax2.transAxes, ha='center', va='center')

        plt.tight_layout()

        return fig, (ax1, ax2), {
            'total_energy': total_energy,
            'mean_energy': mean_energy,
            'max_energy': max_energy,
            'non_zero_count': non_zero_count,
            'total_cells': total_cells,
            'frequency': FREQUENCY_BANDS[freq_idx]
        }

    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
        # Резервный вариант
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.text(0.5, 0.5, f'Ошибка визуализации: {e}', transform=ax1.transAxes, ha='center', va='center')
        ax2.text(0.5, 0.5, 'Нет данных', transform=ax2.transAxes, ha='center', va='center')
        plt.tight_layout()
        return fig, (ax1, ax2), {}

def plot_ray_paths(ray_paths, wall_vertices, microphones, source_position):
    """
    Визуализация путей лучей на отдельном графике
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Рисуем контур помещения
    wall_x = np.append(wall_vertices[0], wall_vertices[0][0])
    wall_y = np.append(wall_vertices[1], wall_vertices[1][0])
    ax.plot(wall_x, wall_y, 'k-', linewidth=3, label='Помещение')

    # Рисуем источник
    ax.scatter(source_position[0], source_position[1], c='red', s=200, marker='*',
               label='Источник', edgecolors='white', linewidth=2, zorder=10)

    # Рисуем микрофоны
    for i, mic in enumerate(microphones):
        mic_x, mic_y = mic['position']
        circle = plt.Circle((mic_x, mic_y), mic['radius'], color='blue', alpha=0.3,
                            label=f'Микрофон {i + 1}' if i == 0 else "")
        ax.add_patch(circle)
        ax.scatter(mic_x, mic_y, c='blue', s=100, marker='s',
                   edgecolors='white', linewidth=2, zorder=5)

    # Собираем все энергии для нормализации цвета
    all_energies = []
    for ray_data in ray_paths:
        # Проверяем длину кортежа и извлекаем энергии
        if len(ray_data) == 4:
            ray_x, ray_y, energies, wall_sequence = ray_data
        elif len(ray_data) == 3:
            ray_x, ray_y, energies = ray_data
        else:
            continue

        if len(energies) > 0:
            # Если энергии - это список векторов (для разных частот), берем среднее
            if isinstance(energies[0], (list, np.ndarray)) and len(energies[0]) > 1:
                avg_energies = [np.mean(e) for e in energies if e is not None]
                all_energies.extend(avg_energies)
            else:
                all_energies.extend(energies)

    if all_energies:
        max_energy = max(all_energies)
        min_energy = min(all_energies)
    else:
        max_energy = 1.0
        min_energy = 0.0

    # Ограничиваем количество отображаемых лучей для лучшей читаемости
    max_rays_to_show = min(360, len(ray_paths))
    step = max(1, len(ray_paths) // max_rays_to_show)

    rays_shown = 0
    for i in range(0, len(ray_paths), step):
        ray_data = ray_paths[i]

        # Извлекаем данные в зависимости от длины кортежа
        if len(ray_data) == 4:
            ray_x, ray_y, energies, wall_sequence = ray_data
        elif len(ray_data) == 3:
            ray_x, ray_y, energies = ray_data
        else:
            continue

        if len(ray_x) < 2:
            continue

        # Вычисляем среднюю энергию для цвета
        if len(energies) > 0:
            if isinstance(energies[0], (list, np.ndarray)) and len(energies[0]) > 1:
                # Если энергии - векторы, берем среднее по частотам
                avg_energy = np.mean([np.mean(e) for e in energies if e is not None])
            else:
                avg_energy = np.mean(energies)
        else:
            avg_energy = 0

        normalized_energy = (avg_energy - min_energy) / (max_energy - min_energy + 1e-8)

        # Цвет от синего (низкая энергия) к красному (высокая энергия)
        color = plt.cm.plasma(normalized_energy)

        # Рисуем путь луча
        ax.plot(ray_x, ray_y, '-', color=color, alpha=0.7, linewidth=1.5)

        # Рисуем точки отражения
        if show_bounce_points and len(ray_x) > 2:
            ax.scatter(ray_x[1:-1], ray_y[1:-1], c='green', s=20, alpha=0.6, marker='o')

        rays_shown += 1
        if rays_shown >= max_rays_to_show:
            break

    # Добавляем colorbar
    if all_energies:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(min_energy, max_energy))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Энергия луча', fontsize=12)

    ax.set_xlabel('X координата (м)', fontsize=12)
    ax.set_ylabel('Y координата (м)', fontsize=12)
    ax.set_title('Визуализация путей звуковых лучей', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Статистика
    stats_text = f'Всего лучей: {len(ray_paths)}\nПоказано лучей: {rays_shown}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig, ax

def analyze_ray_paths(ray_paths):
    """Анализ путей лучей для отладки"""
    print(f"\n=== АНАЛИЗ ПУТЕЙ ЛУЧЕЙ ===")
    multi_segment_rays = 0
    total_segments = 0
    max_segments = 0

    for i, (ray_x, ray_y, energies) in enumerate(ray_paths):
        segments = len(ray_x) - 1
        total_segments += segments
        if segments > 1:
            multi_segment_rays += 1
        if segments > max_segments:
            max_segments = segments

    print(f"Всего лучей: {len(ray_paths)}")
    print(f"Лучей с отражениями: {multi_segment_rays}")
    print(f"Прямых лучей: {len(ray_paths) - multi_segment_rays}")
    print(f"Максимум сегментов в луче: {max_segments}")
    print(f"Среднее сегментов на луч: {total_segments / len(ray_paths):.2f}")



def create_energy_map_numpy(vertices, ray_paths, grid_size=0.2):
    """
    Создает энергетическую карту помещения
    """
    poly_x, poly_y = vertices[0], vertices[1]

    min_x, max_x = np.floor(np.min(poly_x)), np.ceil(np.max(poly_x))
    min_y, max_y = np.floor(np.min(poly_y)), np.ceil(np.max(poly_y))

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)

    energy_matrix = np.zeros((len(y_coords), len(x_coords)))

    polygon = np.column_stack((poly_x, poly_y))
    path = Path(polygon)

    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    inside_mask = path.contains_points(grid_points)
    inside_mask = inside_mask.reshape(xx.shape)

    for ray_path in ray_paths:
        if ray_path is None:
            continue

        ray_x, ray_y, energies = ray_path
        if len(ray_x) < 2:
            continue

        for i in range(len(ray_x) - 1):
            x1, y1 = ray_x[i], ray_y[i]
            x2, y2 = ray_x[i + 1], ray_y[i + 1]
            segment_energy = energies[i] if i < len(energies) else energies[-1]

            energy_matrix = add_energy_to_cells(x1, y1, x2, y2, segment_energy,
                                                x_coords, y_coords, energy_matrix,
                                                grid_size, inside_mask)

    return energy_matrix, x_coords, y_coords, inside_mask


def add_energy_to_cells(x1, y1, x2, y2, energy, x_coords, y_coords, energy_matrix, grid_size, inside_mask):
    """
    Распределение энергии по ячейкам с высокой чувствительностью
    """
    segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if segment_length == 0:
        return energy_matrix

    cells = find_cells_along_ray(x1, y1, x2, y2, x_coords, y_coords, grid_size)

    if cells:
        # Высокое усиление для лучшей видимости
        amplified_energy = energy
        energy_per_cell = amplified_energy / len(cells)

        for i, j in cells:
            if (0 <= i < len(y_coords) and 0 <= j < len(x_coords) and
                    inside_mask[i, j]):
                energy_matrix[i, j] += energy_per_cell

    return energy_matrix


def find_cells_along_ray(x1, y1, x2, y2, x_coords, y_coords, grid_size):
    """
    Находит все ячейки, через которые проходит луч
    """
    cells = set()

    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)

    if length == 0:
        return cells

    # Высокое разрешение для лучшего покрытия
    step = grid_size / 4
    steps = max(1, int(length / step) + 1)

    for t in np.linspace(0, 1, steps * 4):
        x = x1 + t * dx
        y = y1 + t * dy

        i = np.searchsorted(y_coords, y) - 1
        j = np.searchsorted(x_coords, x) - 1

        if 0 <= i < len(y_coords) and 0 <= j < len(x_coords):
            cells.add((i, j))

    return cells
