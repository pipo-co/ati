from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from models.draw_cmd import LineDrawCmd
from models.image import MAX_COLOR, Image, ImageChannelTransformation, ImageTransformation, normalize

from transformations.np_utils import index_matrix
from .sliding import PaddingStrategy, sliding_window, weighted_sum

RHO_RESOLUTION = 125
THETA_RESOLUTION = 91

class Direction(Enum):
    VERTICAL = [
        [ 0,  1,  0],
        [ 0,  1,  0],
        [ 0,  1,  0]
    ]
    NEGATIVE_DIAGONAL = [
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ]
    HORIZONTAL = [
        [ 0,  0,  0],
        [ 1,  1,  1],
        [ 0,  0,  0]
    ]
    POSITIVE_DIAGONAL = [
        [ 0,  0,  1],
        [ 0,  1,  0],
        [ 1,  0,  0]
    ]

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, direction: str) -> 'Direction':
        direction_name = direction.upper()
        if direction_name not in Direction.names():
            raise ValueError(f'"{direction_name.title()}" is not a supported direction')
        return cls[direction_name]

    @classmethod
    def from_angle(cls, angle: int) -> 'Direction':
        if angle == 0:
            return cls.HORIZONTAL
        elif angle == 45:
            return cls.POSITIVE_DIAGONAL
        elif angle == 90:
            return cls.VERTICAL
        elif angle == 135:
            return cls.NEGATIVE_DIAGONAL
        else:
            raise ValueError(f'{angle} is not a valid direction angle')

    @property
    def kernel(self) -> np.ndarray:
        return np.array(self.value)

    # https://stackoverflow.com/a/41506120/12270520
    @staticmethod
    def _outer_slice(x):
        return np.r_[x[0], x[1:-1, -1], x[-1, :0:-1], x[-1:0:-1, 0]]

    @staticmethod
    def _rotate_matrix(x, shift):
        out = np.empty_like(x)
        N = x.shape[0]
        idx = np.arange(x.size).reshape(x.shape)
        for n in range((N + 1) // 2):
            sliced_idx = Direction._outer_slice(idx[n:N - n, n:N - n])
            out.ravel()[sliced_idx] = np.roll(np.take(x, sliced_idx), shift)
        return out

    def align_vertical_kernel(self, kernel: np.ndarray):
        return self._rotate_matrix(kernel, self.value)

class FamousKernel(Enum):
    # x derivative kernel
    PREWITT = [
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ]
    SOBEL = [
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ]
    LAPLACE = [
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ]
    # TODO(tobi): Se cambio el kernel de juliana?? Igual yo no importa
    JULIANA = [
        [ 1,  1, -1],
        [ 1, -2, -1],
        [ 1,  1, -1]
    ]
    SUSAN = [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0]
    ]

    @property
    def kernel(self) -> np.ndarray:
        return np.array(self.value)

def directional_channel(channel: np.ndarray, vertical_kernel: np.ndarray, border_dir: Direction, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    kernel = border_dir.align_vertical_kernel(vertical_kernel)
    return weighted_sum(channel, kernel, padding_str), ImageChannelTransformation()

def high_pass_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    kernel = np.full((kernel_size, kernel_size), -1 / kernel_size)
    kernel[kernel_size // 2, kernel_size // 2] = (kernel_size ** 2 - 1) / kernel_size
    return weighted_sum(channel, kernel, padding_str), ImageChannelTransformation()

def gradient_modulus(img: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    x_channel = weighted_sum(img, kernel, padding_str)
    kernel = np.rot90(kernel, k=-1)
    y_channel = weighted_sum(img, kernel, padding_str)
    return np.sqrt(y_channel ** 2 + x_channel ** 2)

def prewitt_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    return gradient_modulus(channel, FamousKernel.PREWITT.kernel, padding_str), ImageChannelTransformation()

def sobel_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    return gradient_modulus(channel, FamousKernel.SOBEL.kernel, padding_str), ImageChannelTransformation()

def zero_crossing_vertical(data: np.ndarray, threshold: int) -> np.ndarray:
    ans = np.empty(data.shape, dtype=np.bool8)
    # Cambios de signo directos
    ans[:-1] = (data[:-1] * data[1:] < 0) & (np.abs(data[:-1] - data[1:]) > threshold)
    # Cambios con un 0 en el medio
    ans[:-2] |= (data[:-2] * data[2:] < 0) & (data[1:-1] == 0) & (np.abs(data[:-2] - data[2:]) > threshold)
    # Ultimo nunca cruza
    ans[-1] = False

    return ans

def zero_crossing_horizontal(data: np.ndarray, threshold: int) -> np.ndarray:
    ans = np.empty(data.shape, dtype=np.bool8)
    # Cambios de signo directos
    ans[:, :-1] = (data[:, :-1] * data[:, 1:] < 0) & (np.abs(data[:, :-1] - data[:, 1:]) > threshold)
    # Cambios con un 0 en el medio
    ans[:, :-2] |= (data[:, :-2] * data[:, 2:] < 0) & (data[:, 1:-1] == 0) & (np.abs(data[:, :-2] - data[:, 2:]) > threshold)
    # Ultimo nunca cruza
    ans[:, -1] = False

    return ans

def zero_crossing_borders(data: np.ndarray, threshold: int) -> np.ndarray:
    mask = zero_crossing_vertical(data, threshold) | zero_crossing_horizontal(data, threshold)
    ret = np.zeros(data.shape)
    ret[mask] = MAX_COLOR
    return ret

def laplacian_channel(channel: np.ndarray, crossing_threshold: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    # Derivada segunda
    channel = weighted_sum(channel, FamousKernel.LAPLACE.kernel, padding_str)

    # Queremos ver donde se hace 0, pues son los minimos/maximos de la derivada => borde
    return zero_crossing_borders(channel, crossing_threshold), ImageChannelTransformation()

def log_kernel(sigma: float) -> np.ndarray:
    kernel_size = int(sigma * 10 + 1)
    indices = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indices = np.reshape(indices, (kernel_size, kernel_size, 2))
    sum_squared_over_sigma = np.sum(indices**2, axis=2) / sigma**2  # (x^2 + y^2) / sigma^2
    k = (np.sqrt(2 * np.pi) * sigma**3)                             # sqrt(2pi) * sigma^3
    return - ((2 - sum_squared_over_sigma) / k) * np.exp(-sum_squared_over_sigma/2)

def log_channel(channel: np.ndarray, sigma: float, crossing_threshold: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    # Derivada segunda con gauss
    channel = weighted_sum(channel, log_kernel(sigma), padding_str)

    # Queremos ver donde se hace 0, pues son los minimos/maximos de la derivada => borde
    return zero_crossing_borders(channel, crossing_threshold), ImageChannelTransformation()

def susan_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    kernel = FamousKernel.SUSAN.kernel
    sw = sliding_window(channel, np.shape(kernel), padding_str)
    # Expando dimensiones para que sea compatible con el tamaño de la sliding window
    new_channel = np.expand_dims(channel, axis=(2,3))
    absolute_values = np.absolute(sw[:,:]*kernel - new_channel[:,:])
    absolute_values[absolute_values < 15] = 1
    absolute_values[absolute_values >= 15] = 0
    values = 1 - np.sum(absolute_values, axis=(2, 3)) / kernel.size
    values[(values < 0.4) | (values >= 0.85)] = 0
    values[(values >= 0.4) & (values < 0.65)] = 63
    values[(values >= 0.65) & (values < 0.85)] = 255
    return values, ImageChannelTransformation()

def hough_channel(channel: np.ndarray, t0: float, tf: float, t_count: int, r0: float, rf: float, r_count: int, threshold: float, most_fitted_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    p = np.sqrt(2) * np.max(channel.shape)
    
    if (r0 < -p):       r0 = -p
    if (rf > p):        rf = p
    if (t0 < -np.pi):   t0 = -np.pi
    if (tf > np.pi):    tf = np.pi
    
    rho   = np.linspace(r0, rf, r_count)
    theta = np.linspace(t0, tf, t_count)

    indices = np.insert(index_matrix(*channel.shape), 0, -1, axis=2)

    acum = np.empty((r_count, t_count))

    white_points = channel > 0

    for i in range(len(rho)):
        for j in range(len(theta)):
            params = np.hstack((rho[i], np.sin(theta[j]),  np.cos(theta[j])))
            # |rho - y*sin(theta) - x*cos(theta)|
            line = np.abs(np.sum(params * indices, axis=2)) < threshold
            acum[i, j] = np.sum(white_points & line)

    most_fitted_lines = np.argwhere(acum > most_fitted_ratio * acum.max())
    Y = np.transpose(most_fitted_lines)[0]
    X = np.transpose(most_fitted_lines)[1]

    best = np.hstack((rho[Y, None], theta[X, None]))

    lines = list(filter(lambda l: l, (get_border_points(rho, theta, channel.shape) for rho, theta in best)))
    
    return channel, ImageChannelTransformation(lines)  

def get_border_points(rho: float, theta: float, img_shape) -> Optional[LineDrawCmd]:
    if np.isclose(theta, 0):
        return LineDrawCmd(0, rho, img_shape[0]-1, rho)

    max_y = img_shape[0] - 1
    max_x = img_shape[1] - 1

    x_0 = rho / np.cos(theta)
    x_f = (rho - max_y * np.sin(theta)) / np.cos(theta)
    y_0 = rho / np.sin(theta)
    y_f = (rho - max_x * np.cos(theta)) / np.sin(theta)
    
    ans = []

    if x_0 > 0 and x_0 < max_x: ans.append((0, x_0))
    if x_f > 0 and x_f < max_x: ans.append((max_y, x_f))
    if y_0 > 0 and y_0 < max_y: ans.append((y_0, 0))
    if y_f > 0 and y_f < max_y: ans.append((y_f, max_x))

    if len(ans) != 2:
        print(f'Cantidad incorrecta de puntos {len(ans)} para rho:{rho} y theta:{theta}')
        return None
        
    return LineDrawCmd(*ans[0], *ans[1])
    

def canny_drag_borders(gradient_mod: np.ndarray, t1: int, t2: int, max_col: int, max_row: int, row: int, col: int) -> None:
    if t1 < gradient_mod[row, col] < t2:
        # Conectado por un borde de manera 8-conexo
        neighbour = gradient_mod[max(row - 1, 0):min(row + 2, max_row), max(col - 1, 0):min(col + 2, max_col)]
        gradient_mod[row, col] = MAX_COLOR if np.amax(neighbour) == MAX_COLOR else 0

# Asume que ya fue paso por un filtro gaussiano
def canny_channel(channel: np.ndarray, t1: int, t2: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageChannelTransformation]:
    # Usamos prewitt para derivar
    kernel = FamousKernel.PREWITT.kernel
    dx = weighted_sum(channel, kernel, padding_str)
    kernel = np.rot90(kernel, k=-1)
    dy = weighted_sum(channel, kernel, padding_str)
    gradient_mod = np.sqrt(dy ** 2 + dx ** 2)

    # Calculamos el angulo de la derivada en grados
    d_angle = np.arctan2(dy, dx)
    d_angle[d_angle < 0] += np.pi
    d_angle = np.pi - d_angle
    d_angle = np.rad2deg(d_angle)

    # Discretizamos dicho angulo
    dir_sw = np.empty((*d_angle.shape, *kernel.shape))
    dir_sw[((0 <= d_angle) & (d_angle < 22.5)) | ((157.5 <= d_angle) & (d_angle <= 180))]   = Direction.from_angle(0).kernel
    dir_sw[(22.5 <= d_angle) & (d_angle < 67.5)]                                            = Direction.from_angle(45).kernel
    dir_sw[(67.5 <= d_angle) & (d_angle < 112.5)]                                           = Direction.from_angle(90).kernel
    dir_sw[(112.5 <= d_angle) & (d_angle < 157.5)]                                          = Direction.from_angle(135).kernel

    # Suprimimos los valores que no son maximos
    max_suppression_sw = sliding_window(gradient_mod, kernel.shape, padding_str) * dir_sw
    max_suppression_sw = np.max(max_suppression_sw, axis=(2, 3))
    gradient_mod[max_suppression_sw != gradient_mod] = 0

    # Normalizamos la imagen antes del thresholding
    gradient_mod = normalize(gradient_mod, np.float64)

    # Thresholding con histéresis
    gradient_mod[gradient_mod >= t2] = MAX_COLOR
    gradient_mod[gradient_mod <= t1] = 0

    max_row, max_col = gradient_mod.shape

    # Arrastramos los bordes de manera vertical
    for row in range(max_row):
        for col in range(max_col):
            canny_drag_borders(gradient_mod, t1, t2, max_col, max_row, row, col)

    # Arrastramos los bordes de manera horizontal
    for col in range(max_col):
        for row in range(max_row):
            canny_drag_borders(gradient_mod, t1, t2, max_col, max_row, row, col)

    return gradient_mod, ImageChannelTransformation()

def get_rectangular_boundrie(x: Tuple[int, int], y: Tuple[int, int]) -> np.ndarray:
    upper_line = np.array([(y[0], x) for x in range(x[0], x[1] + 1)])
    bottom_line = np.array([(y[1], x) for x in range(x[0], x[1] + 1)])
    left_line = np.array([(y, x[0]) for y in range(y[0], y[1] +1)])
    right_line = np.array([(y, x[1]) for y in range(y[0], y[1] +1)])

    return np.concatenate((upper_line, right_line, left_line, bottom_line), axis=0)


def get_initial_boundries(x: Tuple[int, int], y: Tuple[int, int], phi_size: int) -> np.ndarray:
    lout = get_rectangular_boundrie(x, y)
    lin = get_rectangular_boundrie((x[0] + 1, x[1] - 1), (y[0] + 1, y[1] - 1))
    phi = np.full((phi_size, phi_size), 3)
    phi[y[0]:y[1], x[0]:x[1]] = 1
    phi[y[0] + 1:y[1] - 1, x[0] + 1:x[1] - 1] = -1
    phi[y[0] + 2:y[1] + 2, x[0] + 2:x[1] - 2] = -3
    return lout, lin, phi

def get_indexes() -> np.ndarray:
    return np.reshape(np.array(list(np.ndindex((3, 3)))) - 3//2, (9, 2))
    
def inBounds(x: int, y: int, shape: Tuple[int, int]) -> Boolean:
    return x >= 0 and x < shape[1] and y >=0 and y < shape[0]

def new_values(phi: np.ndarray, indices:np.ndarray, point: Tuple[int, int], target: int, new_value: int) -> List[Tuple[int, int]]:
    value_list = []
    for index in indices:
        phi_y = point[0] + index[0]
        phi_x = point[1] + index[1]
        if inBounds(phi_x, phi_y, phi.shape) and phi[phi_y, phi_x] == target:
            value_list.append((phi_y, phi_x))
            phi[phi_y, phi_x] = new_value
    return value_list

def active_outline(image: np.ndarray, sigma, lout: np.ndarray, lin: np.ndarray, phi: np.ndarray) -> np.ndarray:
    flag = True
    indices = get_indexes()
    while flag:
        flag = False
        new_lout = []
        new_lin = []
        for point in lout:
            norm_lout = np.linalg.norm(sigma - image[point[0], point[1]])
            if (norm_lout <= 10):
                new_lin.append(point)
                new_lout.extend(new_values(phi, indices, point, 3, 1))
                flag = True
            else:
                new_lout.append(point)
        for point in lin:
            norm_lin = np.linalg.norm(sigma - image[point[0], point[1]])
            if (norm_lin >= 10):
                new_lout.append(point)
                new_lin.extend(new_values(phi, indices, point, -3, -1))
                flag = True
            else:
                new_lin.append(point)
        lout = new_lout
        lin = new_lin
        
    return lout, lin, phi

    

# ******************* Export Functions ********************** #
def directional(image: Image, kernel: FamousKernel, border_dir: Direction, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('directional', directional_channel, vertical_kernel=kernel.kernel, border_dir=border_dir, padding_str=padding_str)

def high_pass(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('high_pass', high_pass_channel, kernel_size=kernel_size, padding_str=padding_str)

def prewitt(image: Image, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('prewitt', prewitt_channel, padding_str=padding_str)

def sobel(image: Image, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('sobel', sobel_channel, padding_str=padding_str)

def laplace(image: Image, crossing_threshold: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('laplace', laplacian_channel, crossing_threshold=crossing_threshold, padding_str=padding_str)

def log(image: Image, sigma: float, crossing_threshold: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('log', log_channel, sigma=sigma, crossing_threshold=crossing_threshold, padding_str=padding_str)

def susan(image: Image, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('susan', susan_channel, padding_str=padding_str)

def hough(image: Image, t0: float, tf: float, t_count: int, r0: float, rf: float, r_count: int, threshold: float, most_fitted_ratio: float) -> Tuple[np.ndarray, ImageTransformation]:    
    return image.apply_over_channels('hough', hough_channel, t0=t0, tf=tf, t_count=t_count, r0=r0, rf=rf, r_count=r_count, threshold=threshold, most_fitted_ratio=most_fitted_ratio)
    
def canny(image: Image, t1: int, t2: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('canny', canny_channel, t1=t1, t2=t2, padding_str=padding_str)

def make_dimension_point(p1: Tuple[int, int], p2: Tuple[int, int], dim: int):
    if p1[dim] > p2[dim]:
        return (p2[dim], p1[dim])
    else:
        return (p1[dim], p2[dim])

def active_outline_first_frame(image: Image, p1: Tuple[int, int], p2: Tuple[int, int]) -> np.ndarray:
    
    x = make_dimension_point(p1, p2, 0)
    y = make_dimension_point(p1, p2, 1)

    if image.channels > 1:
        sigma = np.mean(np.array(image.data[y[0]:y[1], x[0]:x[1]]).reshape((-1, 3)), axis=0)
    else:
        sigma = np.mean(image.data[y[0]:y[1], x[0]:x[1]])

    lout, lin, phi = get_initial_boundries(image, x, y)  

    return active_outline(image, sigma, lout, lin, phi)

def active_outline_middle_frame(image: Image, in_color: int, l_in: np.ndarray, l_out: np.ndarray) -> np.ndarray:
    pass
