import itertools
from typing import Callable, List, Optional

import dearpygui.dearpygui as dpg
import numpy as np

import denoising
import images_repo as img_repo
import interface
import noise
import rng
from denoising import PaddingStrategy, DirectionalOperator
from image_utils import MAX_ANISOTROPIC_ITERATIONS, AnisotropicFunction, Image, anisotropic_diffusion, strip_extension, add_images, sub_images, multiply_images, \
    power_function, negate, to_binary, equalize, ImageFormat, MAX_COLOR, get_extension, normalize, universal_to_binary
from interface_utils import render_error
from noise import NoiseType

# General Items
TR_DIALOG: str = 'tr_dialog'

# Custom Inputs
TR_NAME_INPUT: str              = 'tr_name_input'
TR_IMG_INPUT: str               = 'tr_img_input'
TR_INT_VALUE_SELECTOR: str      = 'tr_int_value_input'
TR_FLOAT_VALUE_SELECTOR: str    = 'tr_float_value_input'
TR_RADIO_BUTTONS: str           = 'tr_radio_buttons'
TR_CHECKBOX: str                = 'tr_checkbox'
TR_INT_TABLE: str               = 'tr_int_table'

TrHandler = Callable[[str], Image]

def build_transformations_menu(image_name: str) -> None:
    with dpg.menu(label='Transform'):
        with dpg.menu(label='Utils'):
            build_tr_menu_item(TR_COPY,     build_copy_dialog, image_name)
            build_tr_menu_item(TR_REFORMAT, build_reformat_dialog, image_name)
            build_tr_menu_item(TR_NORMALIZE, build_normalize_dialog, image_name)
        with dpg.menu(label='Basic'):
            build_tr_menu_item(TR_NEG,      build_neg_dialog, image_name)
            build_tr_menu_item(TR_POW,      build_pow_dialog, image_name)
            build_tr_menu_item(TR_UMBRAL,   build_umbral_dialog, image_name)
            build_tr_menu_item(TR_EQUALIZE, build_equalize_dialog, image_name)
        with dpg.menu(label='Combine'):
            build_tr_menu_item(TR_ADD,      build_add_dialog, image_name)
            build_tr_menu_item(TR_SUB,      build_sub_dialog, image_name)
            build_tr_menu_item(TR_MULT,     build_mult_dialog, image_name)
        with dpg.menu(label='Noise'):
            build_tr_menu_item(TR_NOISE_GAUSS,    build_noise_gauss_dialog, image_name)
            build_tr_menu_item(TR_NOISE_EXP,      build_noise_exp_dialog, image_name)
            build_tr_menu_item(TR_NOISE_RAYLEIGH, build_noise_rayleigh_dialog, image_name)
            build_tr_menu_item(TR_NOISE_SALT,     build_noise_salt_dialog, image_name)
        with dpg.menu(label='Denoise'):
            build_tr_menu_item(TR_DENOISE_MEAN, build_denoise_mean_dialog, image_name)
            build_tr_menu_item(TR_DENOISE_MEDIAN, build_denoise_median_dialog, image_name)
            build_tr_menu_item(TR_DENOISE_WEIGHTED_MEDIAN, build_denoise_weighted_median_dialog, image_name)
            build_tr_menu_item(TR_DENOISE_GAUSS, build_denoise_gauss_dialog, image_name)
            build_tr_menu_item(TR_DENOISE_HIGH, build_denoise_high_dialog, image_name)
            build_tr_menu_item(TR_DIRECTIONAL, build_directional_dialog, image_name)
            build_tr_menu_item(TR_PREWITT,          build_denoise_prewitt_dialog, image_name)
            build_tr_menu_item(TR_SOBEL,            build_denoise_sobel_dialog, image_name)
            build_tr_menu_item(TR_GLOBAL_UMBRAL, build_global_umbral_dialog, image_name)
            build_tr_menu_item(TR_ANISOTROPIC_DIFFUSION, build_anisotropic_diffusion_dialog, image_name)
            build_tr_menu_item(TR_OTSU_THRESHOLD, build_otsu_threshold_dialog, image_name)
            build_tr_menu_item(TR_BILATERAL, build_bilateral_filter_dialog, image_name)
            build_tr_menu_item(TR_LAPLACIAN_BORDER, build_laplacian_border_dialog, image_name)

def build_tr_menu_item(tr_id: str, tr_dialog_builder: Callable[[str], None], image_name: str) -> None:
    dpg.add_menu_item(label=tr_id.title(), user_data=(tr_dialog_builder, image_name), callback=lambda s, ad, ud: ud[0](ud[1]))

@render_error
def execute_transformation(image_name: str, handler: TrHandler) -> None:
    try:
        new_image = handler(image_name)
    finally:
        dpg.delete_item(TR_DIALOG)

    img_repo.persist_image(new_image)
    interface.register_image(new_image)
    interface.render_image_window(new_image.name)

def build_tr_dialog(tr_id: str) -> int:
    return dpg.window(label=f'Apply {tr_id.title()} Transformation', tag=TR_DIALOG, modal=True, no_close=True, pos=interface.CENTER_POS)

def build_tr_dialog_end_buttons(tr_id: str, image_name: str, handle: TrHandler) -> None:
    with dpg.group(horizontal=True):
        dpg.add_button(label='Transform', user_data=(image_name, handle), callback=lambda s, ap, ud: execute_transformation(*ud))
        dpg.add_button(label='Cancel', user_data=tr_id, callback=lambda: dpg.delete_item(TR_DIALOG))

# ******************** Input Builders ********************* #

def unique_image_name(base_name: str, ext: str) -> str:
    if not img_repo.contains_image(base_name + ext):
        return base_name
    for i in itertools.count(start=2):
        name = f'{base_name}_{i}'
        if not img_repo.contains_image(name + ext):
            return name

# Solo puede haber un name input, que (casi) siempre debe estar
def build_tr_name_input(tr_id: str, image_name: str) -> None:
    dpg.add_text('Select New Image Name (no extension)')
    dpg.add_input_text(default_value=unique_image_name(strip_extension(image_name) + f'_{tr_id}', get_extension(image_name)), tag=TR_NAME_INPUT)

def build_tr_value_int_selector(name: str, min_val: int, max_val: int, default_value: int = None, suffix: str = None, step: int = 1, tag: str = TR_INT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select \'{name}\' between {min_val} and {max_val}')
    dpg.add_input_int(min_value=min_val, max_value=max_val, default_value=min_val if default_value is None else default_value, step=step, label=suffix, tag=tag)

def build_tr_value_float_selector(name: str, min_val: float, max_val: float, default_value: float = None, suffix: str = None, tag: str = TR_FLOAT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select \'{name}\' between {min_val} and {max_val}')
    dpg.add_input_float(min_value=min_val, max_value=max_val, default_value=min_val if default_value is None else default_value, label=suffix, tag=tag)
    
def build_tr_percentage_selector(name: str, default_value: int = 20, tag: str = TR_INT_VALUE_SELECTOR) -> None:
    build_tr_value_int_selector(name, 0, 100, default_value=default_value, suffix='%', tag=tag)

def build_tr_radio_buttons(names: List[str], default_value: Optional[str] = None, horizontal: bool = True, tag: str = TR_RADIO_BUTTONS) -> None:
    names = list(map(lambda s: s.replace('_', ' ').title(), names))
    if default_value:
        default_value = default_value.title()
    else:
        default_value = names[0]
    dpg.add_radio_button(items=names, default_value=default_value, horizontal=horizontal, tag=tag)

def build_tr_checkbox(name: str, tag: str = TR_CHECKBOX) -> None:
    dpg.add_checkbox(label=name, tag=tag)
    
def build_tr_img_selector(image_name: str) -> None:
    image_list = list(map(lambda img: img.name, img_repo.get_same_shape_images(image_name)))
    dpg.add_text('Select Another Image to combine')
    dpg.add_listbox(image_list, tag=TR_IMG_INPUT)

def build_tr_input_table(size: int = 3, tag: str = TR_INT_TABLE):
    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()

        for row in range(size):
            with dpg.table_row():
                for col in range(size):
                    with dpg.table_cell():
                        cell_tag = get_table_field_name(tag, row, col)
                        dpg.add_input_int(min_value=1, max_value=50, step=0, tag=cell_tag)

def get_table_field_name(tag: str, row: int, col: int) -> str:
    return f'{tag}_{row}_{col}'

# ******************** Input Getters and Requirements ********************* #

def get_tr_name_value(image: Image) -> str:
    base_name = dpg.get_value(TR_NAME_INPUT)
    if not base_name:
        raise ValueError('A name for the new image must be provided')
    ret = base_name + image.format.to_extension()
    if img_repo.contains_image(ret):
        raise ValueError(f'Another image with name "{ret}" already exists')
    return ret

def get_tr_img_value(img_input: str = TR_IMG_INPUT) -> Image:
    img_name = dpg.get_value(img_input)
    if not img_repo.contains_image(img_name):
        raise ValueError('Selecting a valid image is required for transformation')
    return img_repo.get_image(img_name)

def get_tr_radio_buttons_value(radio_buttons: str = TR_RADIO_BUTTONS) -> str:
    return dpg.get_value(radio_buttons).replace(' ', '_')

def get_tr_checkbox_value(checkbox: str = TR_CHECKBOX) -> bool:
    return dpg.get_value(checkbox)

def get_tr_int_value(int_input: str = TR_INT_VALUE_SELECTOR) -> int:
    return dpg.get_value(int_input)

def get_tr_float_value(float_input: str = TR_FLOAT_VALUE_SELECTOR) -> float:
    return dpg.get_value(float_input)

def get_tr_percentage_value(percentage_input: str = TR_FLOAT_VALUE_SELECTOR) -> int:
    percentage = get_tr_int_value(percentage_input)
    if not 0 <= percentage <= 100:
        raise ValueError('Percentage must be between 0 and 100')
    return percentage

def get_tr_input_table_values(size: int = 3, table_tag: str = TR_INT_TABLE) -> List[List[int]]:
    table_values = []
    for row in range(0, size):
        table_values.append([])
        for col in range(0, size):
            cell_tag = get_table_field_name(table_tag, row, col)
            table_values[row].append(dpg.get_value(cell_tag))

    return table_values

def require_odd(n: int, msg: str) -> int:
    if n % 2 == 0:
        raise ValueError(msg)
    return n

def require_same_shape(img1: Image, img2: Image, msg: str) -> None:
    if img1.shape != img2.shape:
        raise ValueError(msg)

########################################################
# ******************** Utilities ********************* #
########################################################


TR_COPY: str = 'copy'
@render_error
def build_copy_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COPY):
        build_tr_name_input(TR_COPY, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_dialog_end_buttons(TR_COPY, image_name, tr_copy)

def tr_copy(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, image.format, image.data)


TR_REFORMAT: str = 'reformat'
@render_error
def build_reformat_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COPY):
        image = img_repo.get_image(image_name)
        fmts = ImageFormat.values()
        if image.channels > 1:
            # Raw solo funciona para imagenes de un solo canal
            fmts.remove(ImageFormat.RAW.value)
        build_tr_radio_buttons(fmts)
        build_tr_dialog_end_buttons(TR_REFORMAT, image_name, tr_reformat)

def tr_reformat(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_fmt  = ImageFormat.from_str(get_tr_radio_buttons_value().lower())
    new_name = strip_extension(image_name) + new_fmt.to_extension()
    # 2. Procesamos
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, new_fmt, image.data)


TR_NORMALIZE: str = 'normalize'
@render_error
def build_normalize_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COPY):
        build_tr_name_input(TR_NORMALIZE, image_name)
        build_tr_dialog_end_buttons(TR_NORMALIZE, image_name, tr_normalize)

def tr_normalize(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    new_data = normalize(image.data, np.float64)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)

########################################################
# ********************** Basic *********************** #
########################################################


TR_NEG: str = 'neg'
@render_error
def build_neg_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NEG):
        build_tr_name_input(TR_NEG, image_name)
        build_tr_dialog_end_buttons(TR_NEG, image_name, tr_neg)

def tr_neg(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    new_data = negate(image)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_POW: str = 'pow'
@render_error
def build_pow_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_POW):
        build_tr_name_input(TR_POW, image_name)
        build_tr_value_float_selector('gamma', 0, 2)
        build_tr_dialog_end_buttons(TR_POW, image_name, tr_pow)

def tr_pow(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    gamma    = get_tr_float_value()
    # 2. Procesamos
    new_data = power_function(image, gamma)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_UMBRAL: str = 'umbral'
@render_error
def build_umbral_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_UMBRAL):
        build_tr_name_input(TR_UMBRAL, image_name)
        build_tr_value_int_selector('threshold', 0, MAX_COLOR)
        build_tr_dialog_end_buttons(TR_UMBRAL, image_name, tr_umb)

def tr_umb(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    umb      = get_tr_int_value()
    # 2. Procesamos
    new_data = to_binary(image, umb)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_EQUALIZE: str = 'equalize'
@render_error
def build_equalize_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_EQUALIZE):
        build_tr_name_input(TR_EQUALIZE, image_name)
        build_tr_dialog_end_buttons(TR_EQUALIZE, image_name, tr_equalize)

def tr_equalize(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    new_data = equalize(image)
    # 3. Creamos Imagen y finalizamos
    return Image(new_name, image.format, new_data)

########################################################
# ********************** Noise *********************** #
########################################################


TR_NOISE_GAUSS: str = 'gauss'
@render_error
def build_noise_gauss_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_GAUSS):
        build_tr_name_input(TR_NOISE_GAUSS, image_name)
        build_tr_radio_buttons(NoiseType.names(), default_value=NoiseType.ADDITIVE.name)
        build_tr_value_float_selector('sigma', 0, 1, default_value=0.1, tag='sigma')
        build_tr_percentage_selector('noise percentage', tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_GAUSS, image_name, tr_noise_gauss)

def tr_noise_gauss(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sigma       = get_tr_float_value('sigma')
    percentage  = get_tr_percentage_value('percentage')
    noise_type  = NoiseType.from_name(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = noise.pollute(image, lambda size: rng.gaussian(0, sigma, size), noise_type, percentage)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_NOISE_EXP: str = 'exp'
@render_error
def build_noise_exp_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_EXP):
        build_tr_name_input(TR_NOISE_EXP, image_name)
        build_tr_radio_buttons(NoiseType.names(), default_value=NoiseType.MULTIPLICATIVE.name)
        build_tr_value_float_selector('lambda', 1, 5, default_value=3, tag='lambda')
        build_tr_percentage_selector('noise percentage', tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_EXP, image_name, tr_noise_exp)

def tr_noise_exp(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    lam         = get_tr_float_value('lambda')
    percentage  = get_tr_percentage_value('percentage')
    noise_type  = NoiseType.from_name(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = noise.pollute(image, lambda size: rng.exponential(lam, size), noise_type, percentage)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_NOISE_RAYLEIGH: str = 'rayleigh'
@render_error
def build_noise_rayleigh_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_RAYLEIGH):
        build_tr_name_input(TR_NOISE_RAYLEIGH, image_name)
        build_tr_radio_buttons(NoiseType.names(), default_value=NoiseType.MULTIPLICATIVE.name)
        build_tr_value_float_selector('epsilon', 0, 1, default_value=0.6, tag='epsilon')
        build_tr_percentage_selector('noise percentage', tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_RAYLEIGH, image_name, tr_noise_rayleigh)

def tr_noise_rayleigh(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    epsilon     = dpg.get_value('epsilon')
    percentage  = dpg.get_value('percentage')
    noise_type  = NoiseType.from_name(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = noise.pollute(image, lambda size: rng.rayleigh(epsilon, size), noise_type, percentage)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_NOISE_SALT: str = 'salt'
@render_error
def build_noise_salt_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_SALT):
        build_tr_name_input(TR_NOISE_SALT, image_name)
        build_tr_percentage_selector('salt percentage', default_value=5, tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_SALT, image_name, tr_noise_salt)

def tr_noise_salt(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    percentage  = get_tr_percentage_value('percentage')
    # 2. Procesamos
    new_data = noise.salt(image, percentage)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)

########################################################
# ********************* Combine ********************** #
########################################################


TR_ADD: str = 'add'
@render_error
def build_add_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_ADD):
        build_tr_name_input(TR_ADD, image_name)
        build_tr_img_selector(image_name)
        build_tr_dialog_end_buttons(TR_ADD, image_name, tr_add)

def tr_add(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sec_image   = get_tr_img_value()
    require_same_shape(image, sec_image, 'You can only sum images with the same shape')
    # 2. Procesamos
    new_data = add_images(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return Image(new_name, image.format, new_data)


TR_SUB: str = 'sub'
@render_error
def build_sub_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_SUB):
        build_tr_name_input(TR_SUB, image_name)
        build_tr_img_selector(image_name)
        build_tr_dialog_end_buttons(TR_SUB, image_name, tr_sub)

def tr_sub(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sec_image   = get_tr_img_value()
    require_same_shape(image, sec_image, 'You can only sub images with the same shape')
    # 2. Procesamos
    new_data = sub_images(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return Image(new_name, image.format, new_data)


TR_MULT: str = 'mult'
@render_error
def build_mult_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_MULT):
        build_tr_name_input(TR_MULT, image_name)
        build_tr_img_selector(image_name)
        build_tr_dialog_end_buttons(TR_MULT, image_name, tr_mult)

def tr_mult(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sec_image   = get_tr_img_value()
    require_same_shape(image, sec_image, 'You can only multiply images with the same shape')
    # 2. Procesamos
    new_data = multiply_images(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return Image(new_name, image.format, new_data)

########################################################
# ********************* Denoise ********************** #
########################################################


TR_DENOISE_MEAN: str = 'mean'
@render_error
def build_denoise_mean_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_MEAN):
        build_tr_name_input(TR_DENOISE_MEAN, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_MEAN, image_name, tr_denoise_mean)

def tr_denoise_mean(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = denoising.mean(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_DENOISE_MEDIAN: str = 'median'
@render_error
def build_denoise_median_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_MEDIAN):
        build_tr_name_input(TR_DENOISE_MEDIAN, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(denoising.PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_MEDIAN, image_name, tr_denoise_median)

def tr_denoise_median(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = denoising.median(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_DENOISE_WEIGHTED_MEDIAN: str = 'weighted median'
@render_error
def build_denoise_weighted_median_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_WEIGHTED_MEDIAN):
        build_tr_name_input(TR_DENOISE_WEIGHTED_MEDIAN, image_name)
        build_tr_radio_buttons(denoising.PaddingStrategy.names())
        build_tr_input_table()
        build_tr_dialog_end_buttons(TR_DENOISE_WEIGHTED_MEDIAN, image_name, tr_denoise_weighted_median)

def tr_denoise_weighted_median(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel      = np.array(get_tr_input_table_values())
    padding     = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos - Puede ser async
    new_data = denoising.weighted_median(image, kernel, padding)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_DENOISE_GAUSS: str = 'gauss'
@render_error
def build_denoise_gauss_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_GAUSS):
        build_tr_name_input(TR_DENOISE_GAUSS, image_name)
        build_tr_value_float_selector('sigma', 1, 7, default_value=3)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_GAUSS, image_name, tr_denoise_gauss)

def tr_denoise_gauss(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sigma       = get_tr_float_value()
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = denoising.gauss(image, sigma, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_DENOISE_HIGH: str = 'high'
@render_error
def build_denoise_high_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_HIGH):
        build_tr_name_input(TR_DENOISE_HIGH, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_HIGH, image_name, tr_denoise_high)

def tr_denoise_high(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = denoising.high(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_DIRECTIONAL: str = 'directional'
@render_error
def build_directional_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DIRECTIONAL):
        build_tr_name_input(TR_DIRECTIONAL, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_radio_buttons(DirectionalOperator.names(), tag="direction")
        build_tr_checkbox('Alternative Kernel')
        build_tr_dialog_end_buttons(TR_DIRECTIONAL, image_name, tr_directional)

def tr_directional(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    rotations = DirectionalOperator.from_str(get_tr_radio_buttons_value(radio_buttons="direction"))
    kernel = denoising.ALTERNATIVE_DERIVATIVE_KERNEL if get_tr_checkbox_value() else denoising.STANDARD_DERIVATIVE_KERNEL
    # 2. Procesamos
    new_data = denoising.directional(image, kernel, padding_str, rotations.value)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_PREWITT: str = 'prewitt'
@render_error
def build_denoise_prewitt_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_PREWITT):
        build_tr_name_input(TR_PREWITT, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_PREWITT, image_name, tr_prewitt)

def tr_prewitt(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = denoising.prewitt(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_SOBEL: str = 'sobel'
@render_error
def build_denoise_sobel_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_SOBEL):
        build_tr_name_input(TR_SOBEL, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_SOBEL, image_name, tr_sobel)

def tr_sobel(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = denoising.sobel(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_GLOBAL_UMBRAL: str = 'global umbral'
@render_error
def build_global_umbral_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_GLOBAL_UMBRAL):
        build_tr_name_input(TR_GLOBAL_UMBRAL, image_name)
        build_tr_value_int_selector('threshold', 0, MAX_COLOR)
        build_tr_dialog_end_buttons(TR_GLOBAL_UMBRAL, image_name, tr_global_umbral)

def tr_global_umbral(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    umb         = get_tr_int_value()
    # 2. Procesamos
    new_data = universal_to_binary(image, umb)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_ANISOTROPIC_DIFFUSION: str = 'anisotropic'
@render_error
def build_anisotropic_diffusion_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_ANISOTROPIC_DIFFUSION):
        build_tr_name_input(TR_ANISOTROPIC_DIFFUSION, image_name)
        build_tr_value_int_selector('iterations', 0, MAX_ANISOTROPIC_ITERATIONS, default_value=10)
        build_tr_value_int_selector('sigma', 1, 10, default_value=4, tag='sigma')
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_radio_buttons(AnisotropicFunction.names(), tag='function')
        build_tr_dialog_end_buttons(TR_ANISOTROPIC_DIFFUSION, image_name, tr_anisotropic_diffusion)

def tr_anisotropic_diffusion(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    iterations  = get_tr_int_value()
    sigma       = get_tr_int_value(int_input='sigma')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    function = AnisotropicFunction.from_str(get_tr_radio_buttons_value(radio_buttons='function'))
    # 2. Procesamos
    new_data = anisotropic_diffusion(image, iterations, sigma, padding_str, function)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)

TR_OTSU_THRESHOLD: str = 'otsu'
@render_error
def build_otsu_threshold_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_OTSU_THRESHOLD):
        build_tr_name_input(TR_OTSU_THRESHOLD, image_name)
        build_tr_dialog_end_buttons(TR_OTSU_THRESHOLD, image_name, tr_otsu_threshold)

def tr_otsu_threshold(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    # 2. Procesamos
    new_data = denoising.otsu_threshold(image)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_BILATERAL: str = 'bilateral filter'
@render_error
def build_bilateral_filter_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BILATERAL):
        build_tr_name_input(TR_BILATERAL, image_name)
        build_tr_value_int_selector('sigma_space', 0, 10, default_value=2, tag='sigma_space')
        build_tr_value_int_selector('sigma_intensity', 0, 20, default_value=3, tag='sigma_intensity')
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_BILATERAL, image_name, tr_bilateral_filter)

def tr_bilateral_filter(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sigma_space  = get_tr_int_value(int_input='sigma_space')
    sigma_intensity       = get_tr_int_value(int_input='sigma_intensity')

    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data = denoising.bilateral_filter(image, sigma_space, sigma_intensity, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_LAPLACIAN_BORDER: str = 'laplacian'
@render_error
def build_laplacian_border_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_OTSU_THRESHOLD):
        build_tr_name_input(TR_OTSU_THRESHOLD, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_value_int_selector('crossing threshold', 0, MAX_COLOR, default_value=100, tag='thresh')
        build_tr_dialog_end_buttons(TR_OTSU_THRESHOLD, image_name, tr_laplacian_border)

def tr_laplacian_border(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    crossing_threshold = get_tr_int_value('thresh')
    # 2. Procesamos
    new_data = denoising.laplace(image, crossing_threshold, kernel_size, padding_str)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)
