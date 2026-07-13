# AcousticsGPU — GPU-ускоренная библиотека для акустического моделирования.
#
# Модули:
#     - raytracing: трассировка лучей (поздняя реверберация)
#     - imagesource: метод зеркальных источников (ранние отражения)
#
# Пример использования:
#     from acoustics_gpu import EnergyMapOptimizer, MATERIAL_LIBRARY
#     optimizer = EnergyMapOptimizer(vertices, materials_per_wall=['ковёр_табличный']*4)
#     energy_maps, x, y, mask = optimizer.create_energy_map_ultra_fast(ray_paths)


__version__ = "1.0.0"
__author__ = "Evg5xa"

# Общие компоненты
from .common.materials import (
    MATERIAL_LIBRARY,
    FREQUENCY_BANDS,
    N_FREQUENCIES,
    FREQ_LABELS,
    SPEED_OF_SOUND,
)
from .common.utils import (
    device,
    get_float_input,
    get_int_input,
    get_bool_input,
)

# ===== Ray Tracing =====
from .raytracing import (
    EnergyMapOptimizer,
    trace_ray_spectral,
    plot_spectral_energy_maps,
    plot_frequency_response,
    visualize_energy_map_gpu_optimized,
    plot_ray_paths,
)

# ===== Image Source =====
from .imagesource import (
    get_virtual_sources_torch,
    calculate_paths_and_energies_torch,
    fit_energy_curves_torch,
    draw_room_and_sources_torch,
    validate_input_coefficients,
)

__all__ = [
    # Общее
    "MATERIAL_LIBRARY",
    "FREQUENCY_BANDS",
    "N_FREQUENCIES",
    "FREQ_LABELS",
    "SPEED_OF_SOUND",
    "device",
    "get_float_input",
    "get_int_input",
    "get_bool_input",
    # Ray Tracing
    "EnergyMapOptimizer",
    "trace_ray_spectral",
    "plot_spectral_energy_maps",
    "plot_frequency_response",
    "visualize_energy_map_gpu_optimized",
    "plot_ray_paths",
    # Image Source
    "get_virtual_sources_torch",
    "calculate_paths_and_energies_torch",
    "fit_energy_curves_torch",
    "draw_room_and_sources_torch",
    "validate_input_coefficients",
]