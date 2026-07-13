# Image Source модуль — моделирование ранних отражений.
#
# Содержит:
#     - get_virtual_sources_torch: генерация виртуальных источников на GPU
#     - calculate_paths_and_energies_torch: расчёт путей и энергий
#     - draw_room_and_sources_torch: визуализация комнаты и источников
#     - fit_energy_curves_torch: аппроксимация кривых затухания
#
# Пример:
#     from acoustics_gpu.imagesource import draw_room_and_sources_torch
#     draw_room_and_sources_torch(room_width, room_height, source, receiver, max_order, coeffs)


from .core import (
    get_virtual_sources_torch,
    find_wall_intersections_torch,
    calculate_paths_and_energies_torch,
    fit_energy_curves_torch,
    validate_input_coefficients,
    get_user_input,
)

from .visualize import (
    draw_room_and_sources_torch,
    plot_reflection_path,
)

__all__ = [
    # Core
    "get_virtual_sources_torch",
    "find_wall_intersections_torch",
    "calculate_paths_and_energies_torch",
    "fit_energy_curves_torch",
    "validate_input_coefficients",
    "get_user_input",
    # Visualize
    "draw_room_and_sources_torch",
    "plot_reflection_path",
]