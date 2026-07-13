import torch
import numpy as np
import math
import random
from matplotlib.path import Path
from ..common.materials import FREQUENCY_BANDS, N_FREQUENCIES, MATERIAL_LIBRARY, SPEED_OF_SOUND
from ..common.utils import device, calculate_total_distance

class EnergyMapOptimizer:
    def __init__(self, vertices, grid_size=0.2, materials_per_wall=None):
        self.grid_size = grid_size
        self.frequencies = torch.tensor(FREQUENCY_BANDS, dtype=torch.float32, device=device)
        self.n_frequencies = N_FREQUENCIES

        # Если материалы не заданы, используем бетон по умолчанию
        if materials_per_wall is None:
            materials_per_wall = ['бетон'] * 4

        self.materials = materials_per_wall
        self.setup_grid(vertices)
        self.setup_material_properties(materials_per_wall)
        self.precompute_geometry()

    def setup_material_properties(self, materials_per_wall):
        """Настройка частотно-зависимых свойств материалов"""
        # Создаем тензоры для коэффициентов поглощения и дифракции
        absorption_matrix = []
        diffusion_matrix = []

        for material in materials_per_wall:
            if material in MATERIAL_LIBRARY:
                absorption_matrix.append(MATERIAL_LIBRARY[material]['absorption'])
                diffusion_matrix.append(MATERIAL_LIBRARY[material]['diffusion'])
            else:
                # Если материал не найден, используем бетон
                print(f"Предупреждение: материал '{material}' не найден, используется бетон")
                absorption_matrix.append(MATERIAL_LIBRARY['бетон']['absorption'])
                diffusion_matrix.append(MATERIAL_LIBRARY['бетон']['diffusion'])

        # Преобразуем в тензоры [n_walls, n_frequencies]
        self.absorption_coeffs = torch.tensor(absorption_matrix, dtype=torch.float32, device=device)
        self.diffusion_coeffs = torch.tensor(diffusion_matrix, dtype=torch.float32, device=device)

        # Коэффициенты отражения (1 - поглощение - дифракция)
        self.reflection_coeffs = 1.0 - self.absorption_coeffs - self.diffusion_coeffs

        print(f"Загружены частотные характеристики для {len(materials_per_wall)} стен")
        print(f"Диапазон частот: {FREQUENCY_BANDS[0]}-{FREQUENCY_BANDS[-1]} Гц")

    def setup_grid(self, vertices):
        """Настройка сетки и предварительные вычисления"""
        poly_x, poly_y = vertices[0], vertices[1]

        # Определяем границы комнаты
        self.min_x, self.max_x = np.floor(np.min(poly_x)), np.ceil(np.max(poly_x))
        self.min_y, self.max_y = np.floor(np.min(poly_y)), np.ceil(np.max(poly_y))

        # Создаем координаты сетки
        self.x_coords = np.arange(self.min_x, self.max_x + self.grid_size, self.grid_size)
        self.y_coords = np.arange(self.min_y, self.max_y + self.grid_size, self.grid_size)

        # Создаем полигон для проверки принадлежности точек
        polygon = np.column_stack((poly_x, poly_y))
        path = Path(polygon)

        # Создаем сетку точек
        xx, yy = np.meshgrid(self.x_coords, self.y_coords)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Маска точек внутри помещения
        self.inside_mask = path.contains_points(grid_points)
        self.inside_mask = self.inside_mask.reshape(xx.shape)

        # Переносим данные на GPU
        self.x_coords_tensor = torch.tensor(self.x_coords, dtype=torch.float32, device=device)
        self.y_coords_tensor = torch.tensor(self.y_coords, dtype=torch.float32, device=device)
        self.inside_mask_tensor = torch.tensor(self.inside_mask, dtype=torch.bool, device=device)

    def precompute_geometry(self):
        """Предварительные вычисления геометрии на GPU"""
        # Координаты центров ячеек
        self.cell_centers_x = self.x_coords_tensor + self.grid_size / 2
        self.cell_centers_y = self.y_coords_tensor + self.grid_size / 2

        # Сетка центров ячеек
        self.centers_xx, self.centers_yy = torch.meshgrid(
            self.cell_centers_x, self.cell_centers_y, indexing='xy'
        )

        # Предварительно вычисленные константы
        self.influence_radius = self.grid_size * 2.0
        self.influence_radius_sq = self.influence_radius ** 2

    def create_energy_map_ultra_fast(self, ray_paths):
        """
        Создание энергетической карты с учётом частотных характеристик
        Возвращает карту для каждой частоты
        """
        # Инициализируем энергетические матрицы для каждой частоты
        energy_maps = torch.zeros(
            (self.n_frequencies, len(self.y_coords), len(self.x_coords)),
            device=device
        )

        # Собираем все сегменты для пакетной обработки
        all_segments_data = self._collect_all_segments(ray_paths)

        if not all_segments_data:
            return energy_maps, self.x_coords_tensor, self.y_coords_tensor, self.inside_mask_tensor

        # Обрабатываем сегменты с учётом частот
        energy_maps = self._process_segments_spectral(all_segments_data, energy_maps)

        return energy_maps, self.x_coords_tensor, self.y_coords_tensor, self.inside_mask_tensor

    def _collect_all_segments(self, ray_paths):
        """Сбор всех сегментов лучей с информацией о стенах"""
        segments_data = []

        for ray_path in ray_paths:
            if ray_path is None:
                continue

            # Теперь ray_path содержит информацию о стенах
            if len(ray_path) == 4:  # ray_x, ray_y, energies, wall_sequence
                ray_x, ray_y, energies, wall_sequence = ray_path
            else:
                ray_x, ray_y, energies = ray_path
                wall_sequence = []  # Для обратной совместимости

            if len(ray_x) < 2:
                continue

            for i in range(len(ray_x) - 1):
                x1, y1 = ray_x[i], ray_y[i]
                x2, y2 = ray_x[i + 1], ray_y[i + 1]

                # Получаем энергию для этого сегмента (вектор по частотам)
                if i < len(energies):
                    segment_energy = energies[i]
                else:
                    segment_energy = energies[-1]

                # Получаем стену, от которой отразился луч в этом сегменте
                wall_idx = wall_sequence[i] if i < len(wall_sequence) else -1

                # Проверяем, есть ли ещё энергия (на любой частоте)
                if isinstance(segment_energy, (list, np.ndarray, torch.Tensor)):
                    if max(segment_energy) > 0.001:
                        segments_data.append((x1, y1, x2, y2, segment_energy, wall_idx))
                elif segment_energy > 0.001:
                    # Если скаляр - дублируем для всех частот
                    energy_vector = [segment_energy] * self.n_frequencies
                    segments_data.append((x1, y1, x2, y2, energy_vector, wall_idx))

        return segments_data

    def _process_segments_spectral(self, segments_data, energy_maps):
        """Обработка сегментов с частотным разрешением"""
        batch_size = min(200, len(segments_data))  # Меньший размер пакета из-за частот

        for batch_start in range(0, len(segments_data), batch_size):
            batch_end = min(batch_start + batch_size, len(segments_data))
            batch_segments = segments_data[batch_start:batch_end]

            # Векторизованная обработка пакета для всех частот
            energy_maps = self._process_batch_spectral(batch_segments, energy_maps)

        return energy_maps

    def _process_batch_spectral(self, batch_segments, energy_maps):
        """Векторизованная обработка пакета с учётом частот"""
        n_segments = len(batch_segments)

        # Конвертируем сегменты в тензоры
        x1_list, y1_list, x2_list, y2_list, energy_lists, wall_indices = zip(*batch_segments)

        x1 = torch.tensor(x1_list, dtype=torch.float32, device=device)
        y1 = torch.tensor(y1_list, dtype=torch.float32, device=device)
        x2 = torch.tensor(x2_list, dtype=torch.float32, device=device)
        y2 = torch.tensor(y2_list, dtype=torch.float32, device=device)

        # Энергии [n_segments, n_frequencies]
        energies = torch.tensor(energy_lists, dtype=torch.float32, device=device)

        # Для каждого сегмента и частоты вычисляем вклад
        for freq_idx in range(self.n_frequencies):
            freq_energies = energies[:, freq_idx]  # [n_segments]

            # Векторизованное вычисление расстояний
            energy_contributions = self._compute_energy_contributions_vectorized(
                x1, y1, x2, y2, freq_energies
            )  # [H, W]

            # Применяем маску помещения
            valid_contributions = energy_contributions * self.inside_mask_tensor

            # Добавляем к карте для этой частоты
            energy_maps[freq_idx] += valid_contributions

        return energy_maps

    def _compute_energy_contributions_vectorized(self, x1, y1, x2, y2, energies):
        """Векторизованное вычисление вкладов энергии"""
        # Расширяем размерности для broadcasting
        x1_exp = x1.unsqueeze(0).unsqueeze(0)
        y1_exp = y1.unsqueeze(0).unsqueeze(0)
        x2_exp = x2.unsqueeze(0).unsqueeze(0)
        y2_exp = y2.unsqueeze(0).unsqueeze(0)

        centers_xx_exp = self.centers_xx.unsqueeze(-1)
        centers_yy_exp = self.centers_yy.unsqueeze(-1)

        # Векторизованное вычисление расстояний
        distances_sq = self._distance_to_segment_squared_vectorized(
            centers_xx_exp, centers_yy_exp, x1_exp, y1_exp, x2_exp, y2_exp
        )

        # Вычисляем веса влияния
        weights = torch.where(
            distances_sq <= self.influence_radius_sq,
            1.0 - torch.sqrt(distances_sq) / self.influence_radius,
            torch.tensor(0.0, device=device)
        )

        # Распределяем энергию
        energies_exp = energies.unsqueeze(0).unsqueeze(0)
        energy_contributions = weights * energies_exp

        # Суммируем вклады от всех сегментов
        total_energy = torch.sum(energy_contributions, dim=2)

        return total_energy

    def _distance_to_segment_squared_vectorized(self, px, py, x1, y1, x2, y2):
        """Векторизованное вычисление квадрата расстояния"""
        dx = x2 - x1
        dy = y2 - y1

        fx = px - x1
        fy = py - y1

        segment_length_sq = dx * dx + dy * dy
        segment_length_sq = torch.where(
            segment_length_sq == 0,
            torch.tensor(1e-10, device=device),
            segment_length_sq
        )

        projection = (fx * dx + fy * dy) / segment_length_sq
        projection = torch.clamp(projection, 0, 1)

        closest_x = x1 + projection * dx
        closest_y = y1 + projection * dy

        distance_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2

        return distance_sq


def reflect_ray(normal_x, normal_y, incident_sin, incident_cos):
    normal_length = math.sqrt(normal_x ** 2 + normal_y ** 2)
    if normal_length > 0:
        normal_x /= normal_length
        normal_y /= normal_length

    dot_product = incident_cos * normal_x + incident_sin * normal_y
    reflect_cos = incident_cos - 2 * dot_product * normal_x
    reflect_sin = incident_sin - 2 * dot_product * normal_y

    reflect_length = math.sqrt(reflect_sin ** 2 + reflect_cos ** 2)
    if reflect_length > 0:
        reflect_sin /= reflect_length
        reflect_cos /= reflect_length

    return reflect_sin, reflect_cos


def line_intersection(p1, p2, p3, p4):
    """Оригинальная CPU версия для надежности"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def point_in_microphone(x, y, microphones):
    for mic in microphones:
        mic_x, mic_y = mic['position']
        distance = math.sqrt((x - mic_x) ** 2 + (y - mic_y) ** 2)
        if distance <= mic['radius']:
            return mic
    return None


def find_next_intersection(x, y, sin_a, cos_a, wall_vertices, microphones, wall_absorption, wall_diffraction):
    ray_end_x = x + cos_a * 1000
    ray_end_y = y + sin_a * 1000

    closest_intersection = None
    closest_distance = float('inf')
    wall_normal = None
    intersection_type = None
    hit_microphone = None
    wall_index = None

    # Проверка пересечения с микрофонами
    for mic in microphones:
        mic_x, mic_y = mic['position']
        radius = mic['radius']

        dx = mic_x - x
        dy = mic_y - y
        proj = dx * cos_a + dy * sin_a

        closest_x = x + cos_a * proj
        closest_y = y + sin_a * proj
        dist_to_center = math.sqrt((closest_x - mic_x) ** 2 + (closest_y - mic_y) ** 2)

        if dist_to_center <= radius:
            if dist_to_center == radius:
                t = proj
                if t >= 0:
                    intersection = (closest_x, closest_y)
                    dist = math.sqrt((intersection[0] - x) ** 2 + (intersection[1] - y) ** 2)
                    if dist > 1e-8 and dist < closest_distance:
                        closest_distance = dist
                        closest_intersection = intersection
                        hit_microphone = mic
                        intersection_type = 'microphone'
            else:
                chord_half_length = math.sqrt(radius ** 2 - dist_to_center ** 2)
                t1 = proj - chord_half_length
                t2 = proj + chord_half_length

                t = None
                if t1 >= 0 and t2 >= 0:
                    t = min(t1, t2)
                elif t1 >= 0:
                    t = t1
                elif t2 >= 0:
                    t = t2

                if t is not None:
                    intersection = (x + cos_a * t, y + sin_a * t)
                    dist = math.sqrt((intersection[0] - x) ** 2 + (intersection[1] - y) ** 2)
                    if dist > 1e-8 and dist < closest_distance:
                        closest_distance = dist
                        closest_intersection = intersection
                        hit_microphone = mic
                        intersection_type = 'microphone'

    # Проверка пересечения со стенами
    n_walls = wall_vertices.shape[1]
    for i in range(n_walls):
        p1 = (wall_vertices[0, i], wall_vertices[1, i])
        p2 = (wall_vertices[0, (i + 1) % n_walls], wall_vertices[1, (i + 1) % n_walls])

        intersection = line_intersection((x, y), (ray_end_x, ray_end_y), p1, p2)

        if intersection:
            dist = math.sqrt((intersection[0] - x) ** 2 + (intersection[1] - y) ** 2)
            if dist > 1e-8 and dist < closest_distance:
                closest_distance = dist
                closest_intersection = intersection
                intersection_type = 'wall'
                wall_index = i

                wall_dx = p2[0] - p1[0]
                wall_dy = p2[1] - p1[1]
                normal_x = wall_dy
                normal_y = -wall_dx
                wall_normal = (normal_x, normal_y)

    return closest_intersection, wall_normal, intersection_type, hit_microphone, wall_index


def get_diffraction_angle(normal_x, normal_y, current_sin, current_cos):
    """
    Генерирует случайный угол для диффракции
    """
    normal_length = math.sqrt(normal_x ** 2 + normal_y ** 2)
    if normal_length > 0:
        normal_x /= normal_length
        normal_y /= normal_length

    normal_angle = math.atan2(normal_y, normal_x)
    incident_angle = math.atan2(current_sin, current_cos)

    angle_variation = random.uniform(-math.pi / 2, math.pi / 2)
    diffracted_angle = normal_angle + angle_variation
    diffracted_angle = diffracted_angle % (2 * math.pi)

    return math.sin(diffracted_angle), math.cos(diffracted_angle)

def handle_wall_interaction(current_energy, wall_index, wall_absorption, wall_diffraction, normal_x, normal_y,
                            current_sin, current_cos):
    """
    Обработка взаимодействия луча со стеной
    """
    absorption_prob = wall_absorption[wall_index]
    diffraction_prob = wall_diffraction[wall_index]
    reflection_prob = 1.0 - absorption_prob - diffraction_prob

    rand_val = random.random()

    if rand_val < absorption_prob:
        return 'absorbed', 0.0, 0.0, 0.0
    elif rand_val < absorption_prob + diffraction_prob:
        diffract_sin, diffract_cos = get_diffraction_angle(normal_x, normal_y, current_sin, current_cos)
        return 'diffracted', current_energy * (1.0 - energyLoss), diffract_sin, diffract_cos
    else:
        reflect_sin, reflect_cos = reflect_ray(normal_x, normal_y, current_sin, current_cos)
        return 'reflected', current_energy * (1.0 - energyLoss), reflect_sin, reflect_cos

def trace_ray_spectral(x, y, sin_a, cos_a, initial_energy_vector, wall_vertices, max_bounces,
                       microphones, ray_id, wall_absorption, wall_diffraction, reflection_coeffs=None):
    """
    Трассировка луча с учётом частотных характеристик
    """
    current_x, current_y = x, y
    current_sin, current_cos = sin_a, cos_a

    # Убеждаемся, что initial_energy_vector - тензор на GPU
    if not torch.is_tensor(initial_energy_vector):
        current_energy = torch.tensor(initial_energy_vector, device=device, dtype=torch.float32)
    else:
        current_energy = initial_energy_vector.clone().to(device)

    ray_path_x = [current_x]
    ray_path_y = [current_y]
    energies = [current_energy.cpu().numpy()]
    wall_sequence = []
    bounce_count = 0
    total_distance = 0.0

    for bounce in range(max_bounces):
        # Проверяем энергию
        if torch.max(current_energy) < 0.001:
            break

        # Находим следующее пересечение
        intersection_data = find_next_intersection(
            current_x, current_y, current_sin, current_cos, wall_vertices, microphones,
            wall_absorption, wall_diffraction)

        if intersection_data is None:
            break

        intersection, wall_normal, intersection_type, hit_microphone, wall_index = intersection_data

        if intersection is None:
            break

        # Вычисляем расстояние
        segment_distance = math.sqrt((intersection[0] - current_x) ** 2 + (intersection[1] - current_y) ** 2)
        total_distance += segment_distance

        # Без потерь от расстояния — энергия зависит только от отражений
        segment_energy = current_energy

        # Добавляем точку в путь луча
        ray_path_x.append(intersection[0])
        ray_path_y.append(intersection[1])

        # Сохраняем энергию ЭТОГО сегмента
        energies.append(segment_energy.cpu().numpy())

        # Проверка микрофона
        if intersection_type == 'microphone' and hit_microphone:
            arrival_time = total_distance / speed_of_sound

            if 'frequency_response' not in hit_microphone:
                hit_microphone['frequency_response'] = np.zeros(N_FREQUENCIES)

            segment_energy_np = segment_energy.cpu().numpy()
            hit_microphone['frequency_response'] += segment_energy_np
            hit_microphone['absorbed_energy'] += float(torch.sum(segment_energy))
            hit_microphone['ray_count'] += 1
            hit_microphone['energy_history'].append(segment_energy_np)
            hit_microphone['time_history'].append(arrival_time)
            hit_microphone['distance_history'].append(total_distance)
            hit_microphone['bounce_history'].append(bounce_count)

            wall_sequence.append(wall_index if wall_index is not None else -1)
            break

        # Обновляем позицию
        current_x, current_y = intersection

        # Обработка стены
        if intersection_type == 'wall' and wall_index is not None and wall_normal is not None:

            # ПРИМЕНЯЕМ КОЭФФИЦИЕНТЫ ОТРАЖЕНИЯ - это ключевой момент!
            if reflection_coeffs is not None and wall_index < len(reflection_coeffs):
                # Получаем коэффициенты для этой стены
                wall_reflection = reflection_coeffs[wall_index]

                # Убеждаемся, что это тензор
                if not torch.is_tensor(wall_reflection):
                    wall_reflection = torch.tensor(wall_reflection, device=device)

                # УМНОЖАЕМ - здесь появляется частотная зависимость
                new_energy = segment_energy * wall_reflection

                # Диагностика для первого луча
                if ray_id == 0 and bounce == 0:
                    print(f"\n=== ПЕРВОЕ ОТРАЖЕНИЕ ЛУЧА {ray_id} ===")
                    print(f"  Энергия ДО: {segment_energy.cpu().numpy()}")
                    print(f"  Коэфф стены {wall_index}: {wall_reflection.cpu().numpy()}")
                    print(f"  Энергия ПОСЛЕ: {new_energy.cpu().numpy()}")
            else:
                new_energy = segment_energy * (1.0 - energyLoss)

            # Определяем вероятности (усредняем по частотам)
            if hasattr(wall_absorption[wall_index], '__len__'):
                avg_absorption = float(torch.mean(wall_absorption[wall_index]).item())
                avg_diffraction = float(torch.mean(wall_diffraction[wall_index]).item())
            else:
                avg_absorption = float(wall_absorption[wall_index])
                avg_diffraction = float(wall_diffraction[wall_index])

            # Случайный выбор типа взаимодействия
            rand_val = random.random()
            wall_sequence.append(wall_index)

            if rand_val < avg_absorption:
                # ПОГЛОЩЕНИЕ - энергия обнуляется
                current_energy = torch.zeros_like(new_energy)
                break
            elif rand_val < avg_absorption + avg_diffraction:
                # ДИФРАКЦИЯ - меняем направление, СОХРАНЯЕМ частотную энергию
                diffract_sin, diffract_cos = get_diffraction_angle(
                    wall_normal[0], wall_normal[1], current_sin, current_cos)
                current_sin, current_cos = diffract_sin, diffract_cos
                current_energy = new_energy  # ← ВАЖНО! Сохраняем частотно-зависимую энергию
                bounce_count += 1
            else:
                # ОТРАЖЕНИЕ - меняем направление, СОХРАНЯЕМ частотную энергию
                reflect_sin, reflect_cos = reflect_ray(
                    wall_normal[0], wall_normal[1], current_sin, current_cos)
                current_sin, current_cos = reflect_sin, reflect_cos
                current_energy = new_energy  # ← ВАЖНО! Сохраняем частотно-зависимую энергию
                bounce_count += 1

    return ray_path_x, ray_path_y, energies, bounce_count, total_distance, wall_sequence

def calculate_distance_energy_gpu(initial_energy, distance):
    """GPU-ускоренный расчет энергии"""
    if distance_attenuation:
        distance_effect = 1.0 / (1.0 + distance)
        attenuation = distance_loss_linear ** distance
        return initial_energy * distance_effect * attenuation
    else:
        return initial_energy / (1.0 + distance ** 2)


def line_intersection_gpu(p1, p2, p3, p4):
    """GPU-ускоренное вычисление пересечения линий"""
    # Конвертируем в тензоры
    p1_t = torch.tensor(p1, dtype=torch.float32, device=device)
    p2_t = torch.tensor(p2, dtype=torch.float32, device=device)
    p3_t = torch.tensor(p3, dtype=torch.float32, device=device)
    p4_t = torch.tensor(p4, dtype=torch.float32, device=device)

    x1, y1 = p1_t[0], p1_t[1]
    x2, y2 = p2_t[0], p2_t[1]
    x3, y3 = p3_t[0], p3_t[1]
    x4, y4 = p4_t[0], p4_t[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x.item(), y.item())
    return None

# Функции геометрии
def to_angle(sin_a, cos_a):
    angle = math.atan2(sin_a, cos_a)
    if angle < 0:
        angle += 2 * math.pi
    return angle