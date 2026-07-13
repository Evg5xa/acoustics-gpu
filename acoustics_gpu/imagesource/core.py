import torch
import numpy as np
from typing import List, Dict, Tuple
from ..common.materials import SPEED_OF_SOUND
from ..common.utils import device

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Константы
SPEED_OF_SOUND = 343.0  # м/с
FREQUENCIES = [125, 250, 500, 1000, 2000, 4000]
FREQ_LABELS = ['125 Гц', '250 Гц', '500 Гц', '1 кГц', '2 кГц', '4 кГц']
FREQ_COUNT = len(FREQUENCIES)


def get_virtual_sources_torch(source: List[float],
                              room_width: float,
                              room_height: float,
                              max_order: int,
                              reflection_coeffs_freq: Dict[str, List[float]]) -> Dict[str, torch.Tensor]:
    # Преобразуем входные данные в тензоры PyTorch
    source_tensor = torch.tensor(source, dtype=torch.float32, device=device)
    room_width_tensor = torch.tensor(room_width, dtype=torch.float32, device=device)
    room_height_tensor = torch.tensor(room_height, dtype=torch.float32, device=device)

    refl_tensors = {}
    for wall in ['left', 'right', 'bottom', 'top']:
        refl_tensors[wall] = torch.tensor(reflection_coeffs_freq[wall],
                                          dtype=torch.float32, device=device)

    # Генерируем все возможные комбинации n и m
    n_range = torch.arange(-max_order, max_order + 1, device=device)
    m_range = torch.arange(-max_order, max_order + 1, device=device)

    n_grid, m_grid = torch.meshgrid(n_range, m_range, indexing='ij')
    n_flat = n_grid.flatten()
    m_flat = m_grid.flatten()

    # Вычисляем порядки
    orders = torch.abs(n_flat) + torch.abs(m_flat)

    # Фильтруем: порядок > 0 и <= max_order
    valid_mask = (orders > 0) & (orders <= max_order)
    n_valid = n_flat[valid_mask]
    m_valid = m_flat[valid_mask]
    orders_valid = orders[valid_mask]

    # Количество отражений от каждой стены
    left_count = torch.where(n_valid < 0, torch.abs(n_valid), torch.zeros_like(n_valid))
    right_count = torch.where(n_valid > 0, n_valid, torch.zeros_like(n_valid))
    bottom_count = torch.where(m_valid < 0, torch.abs(m_valid), torch.zeros_like(m_valid))
    top_count = torch.where(m_valid > 0, m_valid, torch.zeros_like(m_valid))

    # Преобразуем counts в матрицу для векторных операций
    counts_matrix = torch.stack([left_count, right_count, bottom_count, top_count], dim=1)  # [n_sources, 4]

    # Коэффициенты отражения как матрица [4, 6]
    coeffs_matrix = torch.stack([refl_tensors['left'],
                                 refl_tensors['right'],
                                 refl_tensors['bottom'],
                                 refl_tensors['top']], dim=0)  # [4, 6]

    powered_coeffs = coeffs_matrix[None, :, :] ** counts_matrix[:, :, None]

    # Перемножаем по оси стен (ось 1) для получения [n_sources, 6]
    effective_factors = torch.prod(powered_coeffs, dim=1)

    # Фильтруем источники с нулевой энергией для всех частот
    non_zero_mask = torch.any(effective_factors > 0, dim=1)
    n_valid = n_valid[non_zero_mask]
    m_valid = m_valid[non_zero_mask]
    orders_valid = orders_valid[non_zero_mask]
    effective_factors = effective_factors[non_zero_mask]
    counts_matrix = counts_matrix[non_zero_mask]

    # Вычисляем координаты виртуальных источников
    source_x, source_y = source_tensor
    x_coords = n_valid * room_width_tensor + torch.where(
        n_valid % 2 == 0,
        source_x,
        room_width_tensor - source_x
    )
    y_coords = m_valid * room_height_tensor + torch.where(
        m_valid % 2 == 0,
        source_y,
        room_height_tensor - source_y
    )

    positions = torch.stack([x_coords, y_coords], dim=1)

    return {
        'positions': positions,  # [n_sources, 2]
        'orders': orders_valid,  # [n_sources]
        'energy_factors': effective_factors,  # [n_sources, 6]
        'wall_counts': counts_matrix,  # [n_sources, 4]
        'n': n_valid,  # [n_sources]
        'm': m_valid  # [n_sources]
    }


def find_wall_intersections_torch(p1: torch.Tensor,
                                  p2: torch.Tensor,
                                  room_width: float,
                                  room_height: float) -> List[Tuple]:
    if p1.dim() == 1:
        p1 = p1.unsqueeze(0)
        p2 = p2.unsqueeze(0)

    batch_size = p1.shape[0]
    intersections_list = []

    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]

    room_width_tensor = torch.tensor(room_width, device=device, dtype=torch.float32)
    room_height_tensor = torch.tensor(room_height, device=device, dtype=torch.float32)

    for i in range(batch_size):
        intersections = []

        # Проверка пересечения с левой стеной (x=0)
        if dx[i] != 0:
            t = -p1[i, 0] / dx[i]
            if 0 <= t <= 1:
                y = p1[i, 1] + t * dy[i]
                if 0 <= y <= room_height:
                    intersections.append((0, y.item(), 'left', t.item()))

        # Проверка пересечения с правой стеной (x=room_width)
        if dx[i] != 0:
            t = (room_width_tensor - p1[i, 0]) / dx[i]
            if 0 <= t <= 1:
                y = p1[i, 1] + t * dy[i]
                if 0 <= y <= room_height:
                    intersections.append((room_width, y.item(), 'right', t.item()))

        # Проверка пересечения с нижней стеной (y=0)
        if dy[i] != 0:
            t = -p1[i, 1] / dy[i]
            if 0 <= t <= 1:
                x = p1[i, 0] + t * dx[i]
                if 0 <= x <= room_width:
                    intersections.append((x.item(), 0, 'bottom', t.item()))

        # Проверка пересечения с верхней стеной (y=room_height)
        if dy[i] != 0:
            t = (room_height_tensor - p1[i, 1]) / dy[i]
            if 0 <= t <= 1:
                x = p1[i, 0] + t * dx[i]
                if 0 <= x <= room_width:
                    intersections.append((x.item(), room_height, 'top', t.item()))

        intersections.sort(key=lambda x: x[3])
        intersections_list.append(intersections)

    return intersections_list if batch_size > 1 else intersections_list[0]


def calculate_paths_and_energies_torch(virtual_sources_dict: Dict[str, torch.Tensor],
                                       receiver: List[float],
                                       room_width: float,
                                       room_height: float) -> Dict[str, torch.Tensor]:
    positions = virtual_sources_dict['positions']  # [n_sources, 2]
    energy_factors = virtual_sources_dict['energy_factors']  # [n_sources, 6]
    n_sources = positions.shape[0]

    # Преобразуем receiver в тензор
    receiver_tensor = torch.tensor(receiver, dtype=torch.float32, device=device)
    receiver_batch = receiver_tensor.repeat(n_sources, 1)  # [n_sources, 2]

    # Вычисляем векторы направления
    vectors = receiver_batch - positions  # [n_sources, 2]

    # Вычисляем расстояния напрямую (евклидово расстояние)
    distances = torch.norm(vectors, dim=1)  # [n_sources]

    # Для каждой траектории находим пересечения со стенами
    intersections_list = find_wall_intersections_torch(positions, receiver_batch, room_width, room_height)

    # Вычисляем реальную длину пути с учетом отражений
    real_distances = torch.zeros(n_sources, device=device, dtype=torch.float32)

    for i in range(n_sources):
        if intersections_list[i]:
            # Если есть пересечения, путь состоит из нескольких сегментов
            path_points = [positions[i].cpu().numpy()]
            for inter in intersections_list[i]:
                path_points.append([inter[0], inter[1]])
            path_points.append(receiver)

            # Вычисляем общую длину
            total_dist = 0
            for j in range(len(path_points) - 1):
                p1 = np.array(path_points[j])
                p2 = np.array(path_points[j + 1])
                total_dist += np.linalg.norm(p2 - p1)
            real_distances[i] = total_dist
        else:
            # Если нет пересечений, используем прямое расстояние
            real_distances[i] = distances[i]

    # Вычисляем энергии для всех частот
    # Базовое затухание: 1 / r^2
    base_energy = 1.0 / (real_distances ** 2).unsqueeze(1)  # [n_sources, 1]

    # Умножаем на коэффициенты отражения для каждой частоты
    energies = base_energy * energy_factors  # [n_sources, 6]

    # Время задержки
    times = real_distances / SPEED_OF_SOUND  # [n_sources]

    return {
        'distances': real_distances.cpu().numpy(),
        'energies': energies.cpu().numpy(),  # [n_sources, 6]
        'times': times.cpu().numpy(),
        'positions': positions.cpu().numpy(),
        'orders': virtual_sources_dict['orders'].cpu().numpy(),
        'wall_counts': virtual_sources_dict['wall_counts'].cpu().numpy(),
        'intersections': intersections_list
    }




def fit_energy_curves_torch(distances: np.ndarray,
                            energies_matrix: np.ndarray) -> np.ndarray:
    # Преобразуем в тензоры
    distances_tensor = torch.tensor(distances, dtype=torch.float32, device=device)
    energies_tensor = torch.tensor(energies_matrix, dtype=torch.float32, device=device)

    # Используем аналитическое решение: C = mean(E * r^2)
    r_squared = distances_tensor ** 2
    C_values = torch.mean(energies_tensor * r_squared.unsqueeze(1), dim=0)

    return C_values.cpu().numpy()


def validate_input_coefficients(coeffs_dict: Dict[str, List[float]]) -> bool:
    for wall, coeffs in coeffs_dict.items():
        if len(coeffs) != FREQ_COUNT:
            print(f"Ошибка: для стены '{wall}' должно быть {FREQ_COUNT} значений")
            return False
        if any(c < 0 or c > 1 for c in coeffs):
            print(f"Ошибка: коэффициенты для стены '{wall}' должны быть в диапазоне [0, 1]")
            return False
    return True


def get_user_input() -> Tuple[int, Dict[str, List[float]]]:
    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ АКУСТИКИ ПОМЕЩЕНИЯ С GPU УСКОРЕНИЕМ")
    print("=" * 60)

    max_reflection_order = int(input('\nВведите максимальный порядок отражений (рекомендуется 3-5): '))

    print("\nВведите коэффициенты отражения для стен для 6 частотных диапазонов (от 0 до 1):")
    print("Частоты: 125 Гц, 250 Гц, 500 Гц, 1 кГц, 2 кГц, 4 кГц")

    # Для каждой стены вводим 6 значений
    walls = ['левой', 'правой', 'верхней', 'нижней']
    wall_keys = ['left', 'right', 'top', 'bottom']
    coeffs_dict = {}

    for wall_name, wall_key in zip(walls, wall_keys):
        print(f"\n{wall_name.capitalize()} стена:")
        coeffs = []
        for freq in FREQUENCIES:
            while True:
                try:
                    value = float(input(f'  {freq} Гц: '))
                    if 0 <= value <= 1:
                        coeffs.append(value)
                        break
                    else:
                        print("    Ошибка: значение должно быть от 0 до 1")
                except ValueError:
                    print("    Ошибка: введите число")
        coeffs_dict[wall_key] = coeffs

    return max_reflection_order, coeffs_dict


# def main():
#     # Параметры помещения (можно сделать настраиваемыми)
#     room_width = 10.0
#     room_height = 8.0
#     source = [2.0, 3.0]
#     receiver = [5.0, 5.0]
#
#     # Получаем данные от пользователя
#     max_order, reflection_coeffs = get_user_input()
#
#     # Проверяем корректность данных
#     if not validate_input_coefficients(reflection_coeffs):
#         print("\nОшибка ввода данных. Программа завершена.")
#         return
#
#     # Запускаем моделирование
#     try:
#         results = draw_room_and_sources_torch(
#             room_width, room_height, source, receiver,
#             max_order, reflection_coeffs
#         )
#
#         # Вывод итоговой статистики
#         print(f"\n{'=' * 60}")
#         print("ИТОГИ МОДЕЛИРОВАНИЯ:")
#         print(f"Всего виртуальных источников: {results['n_sources']}")
#         print(f"Общее время вычислений: {results['computation_time']:.3f} с")
#         print(f"Средний коэффициент C по частотам: {np.mean(results['C_values']):.4f}")
#
#     except Exception as e:
#         print(f"\nОшибка во время выполнения: {e}")
#         print("Попробуйте уменьшить максимальный порядок отражений.")
#
#
# if __name__ == "__main__":
#     main()