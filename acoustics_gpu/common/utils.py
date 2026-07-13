import warnings
import torch
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

#Игнор
warnings.filterwarnings('ignore', category=UserWarning)

def get_float_input(prompt, default=None):
    if default is not None:
        prompt += f" [по умолчанию: {default}]: "
    else:
        prompt += ": "
    try:
        user_input = input(prompt).strip()
        if user_input == "" and default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return float(user_input)
    except ValueError:
        if default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return 0.0


def get_int_input(prompt, default=None):
    if default is not None:
        prompt += f" [по умолчанию: {default}]: "
    else:
        prompt += ": "
    try:
        user_input = input(prompt).strip()
        if user_input == "" and default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return int(user_input)
    except ValueError:
        if default is not None:
            print(f"Используется значение по умолчанию: {default}")
            return default
        return 0


def get_bool_input(prompt, default=True):
    default_str = "y" if default else "n"
    prompt += f" (y/n) [по умолчанию: {default_str}]: "

    user_input = input(prompt).strip().lower()

    if user_input == "":
        print(f"Используется значение по умолчанию: {default_str}")
        return default

    return user_input in ['y', 'yes', 'да', '1', 'true']

def calculate_total_distance(path_x, path_y):
    """Расчёт общего расстояния, пройденного лучом"""
    total_dist = 0.0
    for i in range(len(path_x) - 1):
        dx = path_x[i + 1] - path_x[i]
        dy = path_y[i + 1] - path_y[i]
        total_dist += math.sqrt(dx ** 2 + dy ** 2)
    return total_dist