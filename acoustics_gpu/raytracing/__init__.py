# Ray Tracing модуль — моделирование поздней реверберации.
#
# Содержит:
#     - EnergyMapOptimizer: класс для GPU-оптимизации и построения энергетических карт
#     - trace_ray_spectral: функция трассировки одного луча с частотной зависимостью
#     - Функции визуализации: plot_spectral_energy_maps, plot_frequency_response и др.
#
# Пример:
#     from acoustics_gpu.raytracing import EnergyMapOptimizer
#     optimizer = EnergyMapOptimizer(vertices, grid_size=0.2)


from .core import (
    EnergyMapOptimizer,
    trace_ray_spectral,
    reflect_ray,
    get_diffraction_angle,
    line_intersection,
    find_next_intersection,
    point_in_microphone,
    handle_wall_interaction,
)

from .visualize import (
    plot_spectral_energy_maps,
    plot_frequency_response,
    visualize_energy_map_gpu_optimized,
    plot_ray_paths,
    analyze_ray_paths,
)

__all__ = [
    # Core
    "EnergyMapOptimizer",
    "trace_ray_spectral",
    "reflect_ray",
    "get_diffraction_angle",
    "line_intersection",
    "find_next_intersection",
    "point_in_microphone",
    "handle_wall_interaction",
    # Visualize
    "plot_spectral_energy_maps",
    "plot_frequency_response",
    "visualize_energy_map_gpu_optimized",
    "plot_ray_paths",
    "analyze_ray_paths",
]