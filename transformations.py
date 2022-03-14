from typing import Callable, List, Optional

import dearpygui.dearpygui as dpg

import images_repo as img_repo
import interface
from denoising import mean, PaddingStrategy, median
from image_utils import Image, pollute_img, salt_img, strip_extension, add_images, sub_images, multiply_images, \
    power_function, get_negative, \
    transform_from_threshold, equalize, ImageFormat
from interface_utils import render_error
from noise import NoiseType

# General Items
TR_DIALOG: str = 'tr_dialog'

# Custom Inputs
TR_NAME_INPUT: str = 'tr_name_input'
TR_IMG_INPUT: str = 'tr_img_input'

TR_INT_VALUE_SELECTOR: str = 'tr_int_value_input'
TR_FLOAT_VALUE_SELECTOR: str = 'tr_float_value_input'

TR_RADIO_BUTTONS: str = 'tr_radio_buttons'

TrHandler = Callable[[str], Image]

def build_transformations_menu(image_name: str) -> None:
    with dpg.menu(label='Transform'):
        with dpg.menu(label='Basic'):
            build_tr_menu_item(TR_COPY, build_copy_dialog, image_name)
            build_tr_menu_item(TR_REFORMAT, build_reformat_dialog, image_name)
            build_tr_menu_item(TR_NEG, build_neg_dialog, image_name)
            build_tr_menu_item(TR_POW, build_pow_dialog, image_name)
            build_tr_menu_item(TR_UMBRAL, build_umbral_dialog, image_name)
            build_tr_menu_item(TR_EQUALIZE, build_equalize_dialog, image_name)
        with dpg.menu(label='Combine'):
            build_tr_menu_item(TR_ADD, build_add_dialog, image_name)
            build_tr_menu_item(TR_SUB, build_sub_dialog, image_name)
            build_tr_menu_item(TR_MULT, build_mult_dialog, image_name)
        with dpg.menu(label='Noise'):
            build_tr_menu_item(TR_GAUSS, build_gauss_dialog, image_name)
            build_tr_menu_item(TR_EXP, build_exp_dialog, image_name)
            build_tr_menu_item(TR_RAYLEIGH, build_rayleigh_dialog, image_name)
            build_tr_menu_item(TR_SALT, build_salt_dialog, image_name)
        with dpg.menu(label='Denoise'):
            build_tr_menu_item(TR_MEAN, build_denoise_mean_dialog, image_name)
            build_tr_menu_item(TR_MEDIAN, build_denoise_median_dialog, image_name)

def build_tr_menu_item(tr_id: str, tr_dialog_builder: Callable[[str], None], image_name: str) -> None:
    dpg.add_menu_item(label=tr_id.capitalize(), user_data=(tr_dialog_builder, image_name),
                      callback=lambda s, ad, ud: ud[0](ud[1]))

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
    return dpg.window(label=f'Apply {tr_id.capitalize()} Transformation', tag=TR_DIALOG, modal=True, no_close=True,
                      pos=interface.CENTER_POS)

def build_tr_name_input(tr_id: str, image_name: str) -> None:
    dpg.add_text('Select New Image Name (no extension)')
    dpg.add_input_text(default_value=strip_extension(image_name) + f'_{tr_id}', tag=TR_NAME_INPUT)

def build_tr_value_int_selector(value: str, min: int, max: int, mtag: str = TR_INT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select {value} value')
    dpg.add_input_int(min_value=min, max_value=max, label=f'pick a value for {value} between {min} and {max}', tag=mtag)

def build_tr_value_float_selector(value: str, min: float, max: float, mtag: str = TR_FLOAT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select {value} value')
    dpg.add_input_float(min_value=min, max_value=max, label=f'pick a value for {value} between {min} and {max}',
                        tag=mtag)

def build_tr_radiobuttons(names: List[str], default_value: Optional[str] = None) -> None:
    names = list(map(str.capitalize, names))
    if default_value:
        default_value = default_value.capitalize()
    else:
        default_value = names[0]
    dpg.add_radio_button(items=names, label='sum', default_value=default_value, tag=TR_RADIO_BUTTONS)

def get_req_tr_name_value(image: Image) -> str:
    base_name = dpg.get_value(TR_NAME_INPUT)
    if not base_name:
        raise ValueError('A name for the new image must be provided')
    ret = base_name + image.format.to_extension()
    if img_repo.contains_image(ret):
        raise ValueError(f'Another image with name "{ret}" already exists')
    return ret

def get_gamma_tr_value() -> float:
    gamma = dpg.get_value(TR_FLOAT_VALUE_SELECTOR)
    if gamma == 1.0:
        raise ValueError('Value cannot be 1')
    return gamma

def build_op_img_selector(image_name: str) -> None:
    image_list = list(map(lambda img: img.name, img_repo.get_same_shape_images(image_name)))
    dpg.add_text('Select Another Image to combine')
    dpg.add_listbox(image_list, tag=TR_IMG_INPUT)

def get_req_tr_img_value() -> Image:
    img_name = dpg.get_value(TR_IMG_INPUT)
    if not img_repo.contains_image(img_name):
        raise ValueError('Selecting a valid image is required for transformation')
    return img_repo.get_image(img_name)

def require_same_shape(img1: Image, img2: Image, msg: str) -> None:
    if img1.shape != img2.shape:
        raise ValueError(msg)

def build_tr_dialog_end_buttons(tr_id: str, image_name: str, handle: TrHandler) -> None:
    with dpg.group(horizontal=True):
        dpg.add_button(label='Transform', user_data=(image_name, handle),
                       callback=lambda s, ap, ud: execute_transformation(*ud))
        dpg.add_button(label='Cancel', user_data=tr_id, callback=lambda: dpg.delete_item(TR_DIALOG))

########################################################
##################### Basic ############################
########################################################

TR_COPY: str = 'copy'
@render_error
def build_copy_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COPY):
        build_tr_name_input(TR_COPY, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_dialog_end_buttons(TR_COPY, image_name, tr_nop)

def tr_nop(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    # 2. Procesamos - Puede ser async
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, image.format, image.data)


TR_REFORMAT: str = 'reformat'
@render_error
def build_reformat_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COPY):
        build_tr_radiobuttons(ImageFormat.values())
        build_tr_dialog_end_buttons(TR_COPY, image_name, tr_reformat)

def tr_reformat(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_fmt = ImageFormat.from_str(dpg.get_value(TR_RADIO_BUTTONS).lower())
    new_name = strip_extension(image_name) + new_fmt.to_extension()
    # 2. Procesamos - Puede ser async
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, new_fmt, image.data)


TR_NEG: str = 'neg'
@render_error
def build_neg_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NEG):
        build_tr_name_input(TR_NEG, image_name)
        build_tr_dialog_end_buttons(TR_NEG, image_name, tr_neg)

def tr_neg(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    # 2. Procesamos - Puede ser async
    new_data = get_negative(image)
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_POW: str = 'pow'
@render_error
def build_pow_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_POW):
        build_tr_name_input(TR_POW, image_name)
        build_tr_value_float_selector('gamma', 0.0, 2.0)
        build_tr_dialog_end_buttons(TR_POW, image_name, tr_pow)

def tr_pow(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    gamma: float = get_gamma_tr_value()
    # 2. Procesamos - Puede ser async
    new_data = power_function(image, gamma)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_UMBRAL: str = 'umbral'
@render_error
def build_umbral_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_UMBRAL):
        build_tr_name_input(TR_UMBRAL, image_name)
        build_tr_value_int_selector('threshold', 0, 255)
        build_tr_dialog_end_buttons(TR_UMBRAL, image_name, tr_umb)

def tr_umb(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    umb: int = dpg.get_value(TR_INT_VALUE_SELECTOR)
    # 2. Procesamos - Puede ser async
    new_data = transform_from_threshold(image, umb)
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
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    # 2. Procesamos
    new_data = equalize(image)
    # 3. Creamos Imagen y finalizamos
    return Image(new_name, image.format, new_data)

########################################################
##################### Noise ############################
########################################################

TR_GAUSS: str = 'gauss'
@render_error
def build_gauss_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_GAUSS):
        build_tr_name_input(TR_GAUSS, image_name)
        build_tr_radiobuttons(['add', 'mult'], 'add')
        build_tr_value_float_selector('sigma', 0, 1, mtag='sigma_value_input')
        build_tr_value_float_selector('percentage', 0, 100, mtag='percentage_value_input')
        build_tr_dialog_end_buttons(TR_GAUSS, image_name, tr_gauss)

def tr_gauss(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    sigma: float = dpg.get_value('sigma_value_input')
    percentage: float = dpg.get_value('percentage_value_input')
    mode: str = dpg.get_value('tr_radio_buttons')
    # 2. Procesamos - Puede ser async
    new_data = pollute_img(image, NoiseType.GAUSS, percentage, sigma, mode)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_EXP: str = 'exp'
@render_error
def build_exp_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_EXP):
        build_tr_name_input(TR_EXP, image_name)
        build_tr_radiobuttons(['add', 'mult'], 'add')
        build_tr_value_float_selector('lambda', 0, 1, mtag='parameter_value_input')
        build_tr_value_float_selector('percentage', 0, 100, mtag='percentage_value_input')
        build_tr_dialog_end_buttons(TR_EXP, image_name, tr_exp)

def tr_exp(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    param: float = dpg.get_value('parameter_value_input')
    percentage: float = dpg.get_value('percentage_value_input')
    mode: str = dpg.get_value('tr_radio_buttons')
    # 2. Procesamos - Puede ser async
    new_data = pollute_img(image, NoiseType.EXP, percentage, param, mode)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_RAYLEIGH: str = 'rayleigh'
@render_error
def build_rayleigh_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_RAYLEIGH):
        build_tr_name_input(TR_RAYLEIGH, image_name)
        build_tr_radiobuttons(['add', 'mult'], 'add')
        build_tr_value_float_selector('rayleigh parameter', 0, 1, mtag='rayleigh_value_input')
        build_tr_value_float_selector('percentage', 0, 100, mtag='percentage_value_input')
        build_tr_dialog_end_buttons(TR_RAYLEIGH, image_name, tr_rayleigh)

def tr_rayleigh(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    param: float = dpg.get_value('rayleigh_value_input')
    percentage: float = dpg.get_value('percentage_value_input')
    mode: str = dpg.get_value('tr_radio_buttons')
    # 2. Procesamos - Puede ser async
    new_data = pollute_img(image, NoiseType.RAYL, percentage, param, mode)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_SALT: str = 'salt'
@render_error
def build_salt_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_SALT):
        build_tr_name_input(TR_SALT, image_name)
        build_tr_value_float_selector('percentage', 0, 100, mtag='percentage_value_input')
        build_tr_dialog_end_buttons(TR_SALT, image_name, tr_salt)

def tr_salt(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    percentage: float = dpg.get_value('percentage_value_input')
    # 2. Procesamos - Puede ser async
    new_data = salt_img(image, percentage)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)

########################################################
##################### Combine ##########################
########################################################

TR_ADD: str = 'add'
@render_error
def build_add_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_ADD):
        build_tr_name_input(TR_ADD, image_name)
        build_op_img_selector(image_name)
        build_tr_dialog_end_buttons(TR_ADD, image_name, tr_add)

def tr_add(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    sec_image = get_req_tr_img_value()
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
        build_op_img_selector(image_name)
        build_tr_dialog_end_buttons(TR_SUB, image_name, tr_sub)

def tr_sub(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    sec_image = get_req_tr_img_value()
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
        build_op_img_selector(image_name)
        build_tr_dialog_end_buttons(TR_MULT, image_name, tr_mult)

def tr_mult(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    sec_image = get_req_tr_img_value()
    require_same_shape(image, sec_image, 'You can only multiply images with the same shape')
    # 2. Procesamos
    new_data = multiply_images(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return Image(new_name, image.format, new_data)


########################################################
################### Denoising ##########################
########################################################

TR_MEAN: str = 'mean'
@render_error
def build_denoise_mean_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_MEAN):
        build_tr_name_input(TR_MEAN, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_value_int_selector('n', 3, 255)
        build_tr_radiobuttons(PaddingStrategy.values())
        build_tr_dialog_end_buttons(TR_MEAN, image_name, tr_mean)


def tr_mean(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    n: int = int(dpg.get_value(TR_INT_VALUE_SELECTOR))
    padding: str = dpg.get_value(TR_RADIO_BUTTONS)
    # 2. Procesamos - Puede ser async
    new_data = image.apply_over_channels(mean, n, padding)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


TR_MEDIAN: str = 'median'
@render_error
def build_denoise_median_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_MEDIAN):
        build_tr_name_input(TR_MEDIAN, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_value_int_selector('n', 3, 255)
        build_tr_radiobuttons(PaddingStrategy.values())
        build_tr_dialog_end_buttons(TR_MEDIAN, image_name, tr_median)

def tr_median(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    n: int = int(dpg.get_value(TR_INT_VALUE_SELECTOR))
    padding: str = dpg.get_value(TR_RADIO_BUTTONS)
    # 2. Procesamos - Puede ser async
    new_data = image.apply_over_channels(median, n, padding)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)
