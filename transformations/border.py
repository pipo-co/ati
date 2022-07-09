import functools
import time
from enum import Enum
from typing import List, Optional, Tuple, Union
from xmlrpc.client import Boolean

import numpy as np
from models.draw_cmd import CircleDrawCmd, DrawCmd, LineDrawCmd, ScatterDrawCmd
from models.image import MAX_COLOR, Image, ImageChannelTransformation, normalize
from transformations.data_models import Measurement
from transformations.denoise import gauss_channel

from transformations.np_utils import gauss_kernel, index_matrix
from .data_models import ActiveOutlineMetrics, LinRange
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

class HarrisR(Enum):
    R1  = functools.partial(lambda ix2, ixy, iy2, k: (ix2 * iy2 - ixy ** 2) - k * (ix2 + iy2) ** 2)
    R2  = functools.partial(lambda ix2, ixy, iy2, k: (ix2 * iy2 - ixy ** 2) - k * (ix2 + iy2) ** 4)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.value(*args, **kwargs)

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, strategy: str) -> 'HarrisR':
        strategy_name = strategy.upper()
        if strategy_name not in HarrisR.names():
            raise ValueError(f'"{strategy_name.title()}" is not a supported Harris function')
        return cls[strategy_name]

def directional_channel(channel: np.ndarray, vertical_kernel: np.ndarray, border_dir: Direction, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = border_dir.align_vertical_kernel(vertical_kernel)
    return weighted_sum(channel, kernel, padding_str)

def high_pass_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = np.full((kernel_size, kernel_size), -1 / kernel_size)
    kernel[kernel_size // 2, kernel_size // 2] = (kernel_size ** 2 - 1) / kernel_size
    return weighted_sum(channel, kernel, padding_str)

def gradient_modulus(img: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    x_channel = weighted_sum(img, kernel, padding_str)
    kernel = np.rot90(kernel, k=-1)
    y_channel = weighted_sum(img, kernel, padding_str)
    return np.sqrt(y_channel ** 2 + x_channel ** 2)

def prewitt_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return gradient_modulus(channel, FamousKernel.PREWITT.kernel, padding_str)

def sobel_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return gradient_modulus(channel, FamousKernel.SOBEL.kernel, padding_str)

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

def laplacian_channel(channel: np.ndarray, crossing_threshold: int, padding_str: PaddingStrategy) -> np.ndarray:
    # Derivada segunda
    channel = weighted_sum(channel, FamousKernel.LAPLACE.kernel, padding_str)

    # Queremos ver donde se hace 0, pues son los minimos/maximos de la derivada => borde
    return zero_crossing_borders(channel, crossing_threshold)

def log_kernel(sigma: float) -> np.ndarray:
    kernel_size = int(sigma * 10 + 1)
    indices = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indices = np.reshape(indices, (kernel_size, kernel_size, 2))
    sum_squared_over_sigma = np.sum(indices**2, axis=2) / sigma**2  # (x^2 + y^2) / sigma^2
    k = (np.sqrt(2 * np.pi) * sigma**3)                             # sqrt(2pi) * sigma^3
    return - ((2 - sum_squared_over_sigma) / k) * np.exp(-sum_squared_over_sigma/2)

def log_channel(channel: np.ndarray, sigma: float, crossing_threshold: int, padding_str: PaddingStrategy) -> np.ndarray:
    # Derivada segunda con gauss
    channel = weighted_sum(channel, log_kernel(sigma), padding_str)

    # Queremos ver donde se hace 0, pues son los minimos/maximos de la derivada => borde
    return zero_crossing_borders(channel, crossing_threshold)

def susan_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
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

    return values

def hough_lines_channel(channel: np.ndarray, theta: np.ndarray, rho: np.ndarray, threshold: float, most_fitted_ratio: float) -> Tuple[np.ndarray, ImageChannelTransformation]:
    indices = np.insert(index_matrix(*channel.shape), 0, -1, axis=2)
    acum = np.empty((rho.size, theta.size))
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
    
    return channel, ImageChannelTransformation({'best': best}, {}, lines)

def hough_circle_channel(channel: np.ndarray, radius: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, threshold: float, most_fitted_ratio: float) -> Tuple[np.ndarray, ImageChannelTransformation]:
    indices = index_matrix(*channel.shape)
    centers = np.stack(np.meshgrid(y_axis, x_axis), -1).reshape((-1, 2))
    acum = np.empty((radius.size, len(centers)))
    white_points = channel > 0

    for i in range(len(radius)):
        for j in range(len(centers)):
            # |rho - (y - b)^2 - (x - a)^2|
            circles = (indices - centers[j]) ** 2
            circles = np.insert(circles, 0, - radius[i] ** 2, axis=2)
            circles = np.abs(np.sum(circles, axis=2)) < threshold
            acum[i, j] = np.sum(white_points & circles)

    most_fitted_lines = np.argwhere(acum > most_fitted_ratio * acum.max())
    Y = np.transpose(most_fitted_lines)[0]
    X = np.transpose(most_fitted_lines)[1]

    best = np.hstack((radius[Y, None], centers[X]))

    overlay = list(filter(lambda l: l, (CircleDrawCmd(r, y, x) for r, y, x in best)))

    return channel, ImageChannelTransformation({'best': best}, {}, overlay)

def get_border_points(rho: float, theta: float, img_shape) -> Optional[LineDrawCmd]:
    if np.isclose(theta, 0):
        return LineDrawCmd(0, round(rho), img_shape[0]-1, round(rho))

    max_y = img_shape[0] - 1
    max_x = img_shape[1] - 1

    x_0 = rho / np.cos(theta)
    x_f = (rho - max_y * np.sin(theta)) / np.cos(theta)
    y_0 = rho / np.sin(theta)
    y_f = (rho - max_x * np.cos(theta)) / np.sin(theta)
    
    ans = []

    if 0 < x_0 < max_x: ans.append((0, x_0))
    if 0 < x_f < max_x: ans.append((max_y, x_f))
    if 0 < y_0 < max_y: ans.append((y_0, 0))
    if 0 < y_f < max_y: ans.append((y_f, max_x))

    if len(ans) != 2:
        raise ValueError(f'Cantidad incorrecta de puntos {len(ans)} para rho:{rho} y theta:{theta}')
        
    return LineDrawCmd(*ans[0], *ans[1])

def canny_drag_borders(gradient_mod: np.ndarray, t1: int, t2: int, max_col: int, max_row: int, row: int, col: int) -> None:
    if t1 < gradient_mod[row, col] < t2:
        # Conectado por un borde de manera 8-conexo
        neighbour = gradient_mod[max(row - 1, 0):min(row + 2, max_row), max(col - 1, 0):min(col + 2, max_col)]
        gradient_mod[row, col] = MAX_COLOR if np.amax(neighbour) == MAX_COLOR else 0

# Asume que ya fue paso por un filtro gaussiano
def canny_channel(channel: np.ndarray, t1: int, t2: int, padding_str: PaddingStrategy) -> np.ndarray:
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

    return gradient_mod
    
def harris_channel(channel: np.ndarray, sigma:int, k: float, threshold: float, r_function: HarrisR, padding_str: PaddingStrategy, with_border: bool) -> Tuple[np.ndarray, ImageChannelTransformation]:
    # Usamos prewitt para derivar
    kernel = FamousKernel.PREWITT.kernel
    dx = weighted_sum(channel, kernel, padding_str)
    kernel = np.rot90(kernel, k=-1)
    dy = weighted_sum(channel, kernel, padding_str)

    ix2 = weighted_sum(np.multiply(dx, dx), gauss_kernel(sigma), padding_str)
    ixy = weighted_sum(np.multiply(dx, dy), gauss_kernel(sigma), padding_str)
    iy2 = weighted_sum(np.multiply(dy, dy), gauss_kernel(sigma), padding_str)

    r = r_function(ix2, ixy, iy2, k)
    r_abs = np.abs(r)

    # Normalizamos la imagen antes del thresholding
    normalized_r = normalize(r_abs, np.float64)

    # Los r cuya magnitud es mayor al threshold
    big_r = normalized_r >= threshold

    corner_points = np.argwhere((r > 0) & big_r)
    border_points = np.argwhere((r < 0) & big_r) if with_border else np.empty((0, 2))

    return channel, ImageChannelTransformation({}, {}, [ScatterDrawCmd(border_points, color=(0, 255, 0)), ScatterDrawCmd(corner_points, color=(255, 0, 0))])

def set_difference(new_values: np.ndarray, removed_values: np.ndarray) -> np.ndarray:
    nrows, ncols = new_values.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [new_values.dtype]}
    C = np.setdiff1d(new_values.copy().view(dtype), removed_values.copy().view(dtype))
    return C.view(new_values.dtype).reshape(-1, ncols)

def get_rectangular_boundary(x: Tuple[int, int], y: Tuple[int, int]) -> List[Tuple[int, int]]:
    boundary = []
    boundary.extend([(y[0], x) for x in range(x[0], x[1] + 1)])
    boundary.extend([(y[1], x) for x in range(x[0], x[1] + 1)])
    boundary.extend([(y, x[0]) for y in range(y[0], y[1] + 1)])
    boundary.extend([(y, x[1]) for y in range(y[0], y[1] + 1)])

    return boundary

def calculate_sum(image: Image, x: Tuple[int, int], y: Tuple[int, int]) -> Union[float, np.ndarray]:
    if image.channels > 1:
        return np.sum(np.array(image.data[y[0]:y[1], x[0]:x[1]]).reshape((-1, 3)), axis=0)
    else:
        return np.sum(image.data[y[0]:y[1], x[0]:x[1]])

def get_initial_boundaries(x: Tuple[int, int], y: Tuple[int, int], shape:Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray]:
    lout = get_rectangular_boundary(x, y)
    lin = get_rectangular_boundary((x[0] + 1, x[1] - 1), (y[0] + 1, y[1] - 1))
    phi = np.full(shape, 3)
    phi[y[0]:y[1] + 1, x[0]:x[1] + 1] = 1
    phi[y[0] + 1:y[1], x[0] + 1:x[1]] = -1
    phi[y[0] + 2:y[1] - 1, x[0] + 2:x[1] - 1] = -3
    return lout, lin, phi

def update_regions(x: Tuple[int, int], y: Tuple[int, int], section_id: int, phi: np.ndarray, psi: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    lout = get_rectangular_boundary(x, y)
    lin = get_rectangular_boundary((x[0] + 1, x[1] - 1), (y[0] + 1, y[1] - 1))

    phi[y[0]:y[1] + 1, x[0]:x[1] + 1] = 1
    phi[y[0] + 1:y[1], x[0] + 1:x[1]] = -1
    phi[y[0] + 2:y[1] - 1, x[0] + 2:x[1] - 1] = -3

    psi[y[0] + 1:y[1], x[0] + 1:x[1]] = section_id

    return lout, lin

def single_switch_io(point: Tuple[int, int], ap_list: List[Tuple[int, int]], del_list: List[Tuple[int, int]], phi: np.ndarray, indices_4: np.ndarray, new_value: int, target: int, section_number: int, psi: np.ndarray) -> Boolean:
    ap_list.append(point)
    phi[point[0], point[1]] = new_value
    new_phi_values(phi, del_list, indices_4, point, target, -new_value)
    return True

def topology_number(pos_list: List[Tuple[int, int]], max_distance: int) -> int:
    ret = 0
    region = []
    next_candidates = []
    while pos_list:
        region.append(pos_list.pop())
        region_size = 1
        while region:
            p1 = region.pop()
            for p2 in pos_list:
                if ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < max_distance:
                    region.append(p2)
                    region_size += 1
                else:
                    next_candidates.append(p2)

            pos_list = next_candidates
            next_candidates = []
        if region_size > 1:
            ret += 1
    return ret

def calculate_tr(point: Tuple[int, int], psi: np.ndarray) -> int:
    indices_8 = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    alpha = []
    bg = []
    obj = []
    for index in indices_8:
        pos_y = point[0] + index[0]
        pos_x = point[1] + index[1]
        if in_bounds(pos_x, pos_y, psi.shape):
            psi_value = psi[pos_y, pos_x]
            if psi_value != 0:
                obj.append((pos_y, pos_x))
                alpha.append(psi_value)
            else:
                bg.append((pos_y, pos_x))

    tobj = topology_number(obj, 2)
    tbg = topology_number(bg, 4)
    alpha = len(set(alpha))
    max_value = max(tobj, tbg)
    if max_value == 0:
        return alpha
    # Puede que sea el menor que no sea 0
    return min(alpha, max_value)

def multiple_switch_io(point: Tuple[int, int], ap_list: List[Tuple[int, int]], del_list: List[Tuple[int, int]], phi: np.ndarray, indices_4: np.ndarray, new_value: int, target: int, section_number: int, psi: np.ndarray) -> Boolean:
    tr = calculate_tr(point, psi)
    if section_number == 0 or tr == 1:
        ap_list.append(point)
        phi[point[0], point[1]] = new_value
        new_phi_values(phi, del_list, indices_4, point, target, -new_value)
        psi[point[0], point[1]] = section_number
        return True
    return False

def in_bounds(x: int, y: int, shape: Tuple[int, int]):
    return 0 <= x < shape[1] and 0 <= y < shape[0]

def new_phi_values(phi: np.ndarray, add_collection: List[Tuple[int, int]], indices: np.ndarray, point: Tuple[int, int], target: int, new_value: int):
    for index in indices:
        phi_y = point[0] + index[0]
        phi_x = point[1] + index[1]
        if in_bounds(phi_x, phi_y, phi.shape):
            if phi[phi_y, phi_x] == target:
                add_collection.append((phi_y, phi_x))
                phi[phi_y, phi_x] = new_value
      
def check_value_state(point: Tuple[int, int], phi: np.ndarray, remove_collection: List[Tuple[int, int]], indices: np.ndarray, condition, new_value: int):
    neighbor: bool = True
    for index in indices:
        phi_y = point[0] + index[0]
        phi_x = point[1] + index[1]
        # Tengo que tener a alguien que sea mi borde contrario al lado
        if in_bounds(phi_x, phi_y, phi.shape) and condition(phi[phi_y, phi_x]):
            neighbor = False
    # si tengo algun vecino que es mi borde contrario y algun vecino que es mi contorno contrario no te borro
    if neighbor:
        phi[point[0], point[1]] = new_value
        remove_collection.append((point[0], point[1]))

def is_exterior(val: int) -> bool:
    return val > 0

def is_interior(val: int) -> bool:
    return val < 0

def check_surroundings(point: Tuple[int, int], phi: np.ndarray, indices: np.ndarray, condition) -> bool:
    for index in indices:
        phi_y = point[0] + index[0]
        phi_x = point[1] + index[1]
        # Tengo que tener a alguien que sea mi borde contrario al lado
        if in_bounds(phi_x, phi_y, phi.shape) and not condition(phi[phi_y, phi_x]):
            return False
    return True

def update_img_channel_transformation(img_channel_transformation: ImageChannelTransformation, active_outline_metrics: List[ActiveOutlineMetrics], phi: np.ndarray, switch, psi:np.ndarray, overlay: List[DrawCmd]):
    img_channel_transformation.internal_results['active_outline_metrics'] = active_outline_metrics
    img_channel_transformation.internal_results['phi'] = phi
    img_channel_transformation.internal_results['psi'] = psi
    img_channel_transformation.internal_results['switch'] = switch
    img_channel_transformation.overlay = overlay

def active_outline_all_channels(image: np.ndarray, threshold:float, sigma_bg: Union[float, np.ndarray], active_outline_metrics: List[ActiveOutlineMetrics], phi: np.ndarray, switch, psi:np.ndarray = None) -> Tuple[np.ndarray, ImageChannelTransformation]:
    indices_4 = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    overlay = []
    img_channel_transformation = ImageChannelTransformation({'threshold': threshold, 'sigma_bg': sigma_bg}, {}, None)
    for section in active_outline_metrics:
        new_lout = []
        new_lin = []
        lout = section.lout
        lin = section.lin
        sigma_obj = section.sigma
        section_number = section.section_number
        flag = True
        while flag:
            flag = False
            for point in lout:
                diff_bg = np.linalg.norm(sigma_bg - image[point[0], point[1]])
                diff_obj = np.linalg.norm(sigma_obj - image[point[0], point[1]])
                norm_lout = np.log(diff_bg / diff_obj)
                if np.all(norm_lout >= threshold) and switch(point, lin, lout, phi, indices_4, -1, 3, section_number, psi):
                    flag = True
                else:
                    new_lout.append(point)
            lout = new_lout
            new_lout = []

            for point in lin:
                if check_surroundings(point, phi, indices_4, is_interior):
                    phi[point[0], point[1]] = -3
                else:
                    new_lin.append(point)
            lin = new_lin
            new_lin = []

            for point in lin:
                norm_lin = np.log(np.linalg.norm(sigma_bg - image[point[0], point[1]]) / np.linalg.norm(sigma_obj - image[point[0], point[1]]))
                if np.all(norm_lin < threshold):
                    flag = True
                    switch(point, lout, lin, phi, indices_4, 1, -3, 0, psi)
                else:
                    new_lin.append(point)
            lin = new_lin
            new_lin = []

            for point in lout:
                if check_surroundings(point, phi, indices_4, is_exterior):
                    phi[point[0], point[1]] = 3
                else:
                    new_lout.append(point)
            lout = new_lout
            new_lout = []

        for i in range(5):
            gaussian_phi = gauss_channel(phi, 1, PaddingStrategy.EDGE)
            for point in lout:
                if gaussian_phi[point[0], point[1]] < 0 and switch(point, lin, lout, phi, indices_4, -1, 3,
                                                                   section_number, psi):
                    pass
                else:
                    new_lout.append(point)
            lout = new_lout
            new_lout = []

            for point in lin:
                if check_surroundings(point, phi, indices_4, is_interior):
                    phi[point[0], point[1]] = -3
                else:
                    new_lin.append(point)
            lin = new_lin
            new_lin = []

            for point in lin:
                if gaussian_phi[point[0], point[1]] > 0:
                    switch(point, lout, lin, phi, indices_4, 1, -3, 0, psi)
                else:
                    new_lin.append(point)
            lin = new_lin
            new_lin = []

            for point in lout:
                if check_surroundings(point, phi, indices_4, is_exterior):
                    phi[point[0], point[1]] = 3
                else:
                    new_lout.append(point)
            lout = new_lout
            new_lout = []

        overlay.append(ScatterDrawCmd(np.asarray(lout), section.lout_color))
        overlay.append(ScatterDrawCmd(np.asarray(lin), section.lin_color))
        section.lin = lin
        section.lout = lout

    update_img_channel_transformation(img_channel_transformation, active_outline_metrics, phi, switch, psi, overlay)

    return image, img_channel_transformation

# ******************* Export Functions ********************** #

def directional(image: Image, kernel: FamousKernel, border_dir: Direction, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(directional_channel, vertical_kernel=kernel.kernel, border_dir=border_dir, padding_str=padding_str)

def high_pass(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(high_pass_channel, kernel_size=kernel_size, padding_str=padding_str)

def prewitt(image: Image, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(prewitt_channel, padding_str=padding_str)

def sobel(image: Image, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(sobel_channel, padding_str=padding_str)

def laplace(image: Image, crossing_threshold: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(laplacian_channel, crossing_threshold=crossing_threshold, padding_str=padding_str)

def log(image: Image, sigma: float, crossing_threshold: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(log_channel, sigma=sigma, crossing_threshold=crossing_threshold, padding_str=padding_str)

def susan(image: Image, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(susan_channel, padding_str=padding_str)

def hough_lines(image: Image, theta: List[int], rho: LinRange, threshold: float, most_fitted_ratio: float) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    theta = np.deg2rad(theta)
    return image.apply_over_channels(hough_lines_channel, theta=theta, rho=rho.to_linspace(), threshold=threshold, most_fitted_ratio=most_fitted_ratio)

def hough_circles(image: Image, radius: LinRange, x_axis: LinRange, y_axis: LinRange, threshold: float, most_fitted_ratio: float) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(hough_circle_channel, radius=radius.to_linspace(), x_axis=x_axis.to_linspace(), y_axis=y_axis.to_linspace(), threshold=threshold, most_fitted_ratio=most_fitted_ratio)

def canny(image: Image, t1: int, t2: int, padding_str: PaddingStrategy) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(canny_channel, t1=t1, t2=t2, padding_str=padding_str)

def harris(image: Image, sigma: int, k: float, threshold: float, function: HarrisR, padding_str: PaddingStrategy, with_border: bool) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return image.apply_over_channels(harris_channel, sigma=sigma, k=k, threshold=threshold, r_function=function, padding_str=padding_str, with_border=with_border)

def active_outline_base(image: Image, selections: List[Tuple[Tuple[int, int], Tuple[int, int]]], threshold: float) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    colours = [((255, 0, 0), (0, 255, 255)), ((0, 0, 255), (255, 255, 0)), ((0, 255, 0), (255, 0, 255))]
    start_time = time.thread_time_ns() // 1000000
    shape = image.data.shape[:2]
    phi = np.full(shape, 3)
    psi = np.zeros(shape)
    bg_sum = np.sum(image.data, axis=(0,1))
    bg_size = np.size(image.data) / image.channels
    metrics = []

    for i, point in enumerate(selections):
        section_id = i + 1
        p1 = point[0]
        p2 = point[1]

        x = p1[1], p2[1]
        y = p1[0], p2[0]

        sum = calculate_sum(image, x, y)
        region_size = (x[1] - x[0]) * (y[1] - y[0])
        sigma_obj = sum / region_size
        lout, lin = update_regions(x, y, section_id, phi, psi)

        metrics.append(ActiveOutlineMetrics(section_id, lout, lin, sigma_obj, colours[i % 3][0], colours[i % 3][1]))
        bg_sum = bg_sum - sum
        bg_size = bg_size - region_size

    sigma_bg = bg_sum / bg_size

    img, tr = active_outline_all_channels(image.data, threshold, sigma_bg, metrics, phi, multiple_switch_io, psi)

    duration = time.thread_time_ns() // 1000000 - start_time
    tr.public_results['duration']           = Measurement(duration, 'ms')
    tr.public_results['total_duration']     = Measurement(duration, 'ms')

    return img, [tr]

def active_outline_inductive(frame: int, prev: Image, current: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    prev_results = prev.last_transformation.channel_transformations[0].all_results()
    inputs = map(prev_results.get, ('threshold', 'sigma_bg', 'active_outline_metrics', 'phi', 'switch', 'psi'))
    start_time = time.thread_time_ns() // 1000000
    img, tr = active_outline_all_channels(current.data, *inputs)

    duration = time.thread_time_ns() // 1000000 - start_time
    tr.public_results['duration']           = Measurement(duration, 'ms')
    tr.public_results['total_duration']     = Measurement(prev_results['total_duration'].magnitude + duration, 'ms')
    tr.public_results['mean_duration']      = Measurement(round(tr.public_results['total_duration'].magnitude / (frame + 1), 2), 'ms')

    return img, [tr]  


