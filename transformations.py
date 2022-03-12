from typing import Dict, Callable

import dearpygui.dearpygui as dpg

import images_repo as img_repo
import interface
from image_utils import Image, strip_extension, add_images, sub_images, multiply_images, power_function, get_negative, \
    transform_from_threshold, pollute_gaussian, equalize
from interface_utils import render_error

# General Items
TR_DIALOG: str = 'tr_dialog'

# Custom Inputs
TR_NAME_INPUT: str = 'tr_name_input'
TR_IMG_INPUT: str = 'tr_img_input'

TR_INT_VALUE_SELECTOR: str = 'tr_int_value_input'
TR_FLOAT_VALUE_SELECTOR: str = 'tr_float_value_input'

TrHandler = Callable[[str], Image]

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
    return dpg.window(label=f'Apply {tr_id.capitalize()} Transformation', tag=TR_DIALOG, modal=True, no_close=True, pos=interface.CENTER_POS)

def build_tr_name_input(tr_id: str, image_name: str) -> None:
    dpg.add_text('Select New Image Name (no extension)')
    dpg.add_input_text(default_value=strip_extension(image_name) + f'_{tr_id}', tag=TR_NAME_INPUT)

def build_tr_value_int_selector(value: str, min: int, max:int, mtag:str = TR_INT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select {value} value')
    dpg.add_input_int( min_value=min, max_value=max, label=f'pick a value for {value} between {min} and {max}', tag=mtag)

def build_tr_value_float_selector(value: str, min: float, max:float, mtag:str = TR_FLOAT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select {value} value')
    dpg.add_input_float( min_value=min, max_value=max, label=f'pick a value for {value} between {min} and {max}', tag=mtag)

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

def get_req_tr_name_value(image: Image) -> str:
    base_name = dpg.get_value(TR_NAME_INPUT)
    if not base_name:
        raise ValueError('A name for the new image must be provided')
    ret = base_name + image.format.to_extension()
    if img_repo.contains_image(ret):
        raise ValueError(f'Another image with name "{ret}" already exists')
    return ret

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
        dpg.add_button(label='Transform', user_data=(image_name, handle), callback=lambda s, ap, ud: execute_transformation(*ud))
        dpg.add_button(label='Cancel', user_data=tr_id, callback=lambda: dpg.delete_item(TR_DIALOG))


TR_NOP: str = 'nop'
@render_error
def build_nop_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOP):
        build_tr_name_input(TR_NOP, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_dialog_end_buttons(TR_NOP, image_name, tr_nop)

def tr_nop(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    # 2. Procesamos - Puede ser async
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, image.format, image.data)\

TR_NEG: str = 'neg'
@render_error
def build_neg_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NEG):
        build_tr_name_input(TR_NEG, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
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
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_value_float_selector('gamma', 0.0, 2.0)
        build_tr_dialog_end_buttons(TR_POW, image_name, tr_pow)

def tr_pow(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    gamma:float = get_gamma_tr_value()
    # 2. Procesamos - Puede ser async
    new_data = power_function(image, gamma)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)

TR_UMB: str = 'umb'
@render_error
def build_umb_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_UMB):
        build_tr_name_input(TR_UMB, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_value_int_selector('threshold', 0, 255)
        build_tr_dialog_end_buttons(TR_UMB, image_name, tr_umb)

def tr_umb(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    umb:int = dpg.get_value(TR_INT_VALUE_SELECTOR)
    # 2. Procesamos - Puede ser async
    new_data = transform_from_threshold(image, umb)
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)

TR_GAUSS: str = 'gauss'
@render_error
def build_gauss_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_UMB):
        build_tr_name_input(TR_UMB, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_value_int_selector('median', 0, 255, mtag='median_value_input')
        build_tr_value_float_selector('sigma', 0, 1, mtag='sigma_value_input')
        build_tr_value_float_selector('percentage', 0, 100, mtag='percentage_value_input')
        build_tr_dialog_end_buttons(TR_GAUSS, image_name, tr_gauss)

def tr_gauss(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_req_tr_name_value(image)
    median:int = dpg.get_value('median_value_input')
    sigma:float = dpg.get_value('sigma_value_input')
    percentage:float = dpg.get_value('percentage_value_input')
    # 2. Procesamos - Puede ser async
    new_data = pollute_gaussian(image, percentage, median, sigma, mode='add')
    # 3. Creamos Imagen
    return Image(new_name, image.format, new_data)


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


TRANSFORMATIONS: Dict[str, Callable[[str], None]] = {
    TR_NOP: build_nop_dialog,
    TR_NEG: build_neg_dialog,
    TR_POW: build_pow_dialog,
    TR_UMB: build_umb_dialog,
    TR_ADD: build_add_dialog,
    TR_SUB: build_sub_dialog,
    TR_MULT: build_mult_dialog,
    TR_EQUALIZE: build_equalize_dialog,
    TR_GAUSS: build_gauss_dialog,
}
