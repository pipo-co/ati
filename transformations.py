from typing import Dict, Callable

import dearpygui.dearpygui as dpg

import images_repo as img_repo
import interface
from image_utils import Image, multiply_images, strip_extension, sum_images, sub_images, multiply_images
from interface_utils import render_error

# General Items
TR_DIALOG: str = 'tr_dialog'

# Custom Inputs
TR_NAME_INPUT: str = 'tr_name_input'
TR_IMG_INPUT: str = 'tr_img_input'

TrHandler = Callable[[str], Image]

@render_error
def execute_transformation(image_name: str, handler: TrHandler) -> None:
    try:
        new_image = handler(image_name)
    except Exception as e:
        dpg.delete_item(TR_DIALOG)
        raise e

    img_repo.persist_image(new_image)
    interface.register_image(new_image)
    interface.render_image_window(new_image.name)

def build_tr_dialog(tr_id: str) -> int:
    return dpg.window(label=f'Apply {tr_id.capitalize()} Transformation', tag=TR_DIALOG, modal=True, no_close=True, pos=interface.CENTER_POS)

def build_tr_name_input(tr_id: str, image_name: str) -> None:
    dpg.add_text('Select New Image Name (no extension)')
    dpg.add_input_text(default_value=strip_extension(image_name) + f'_{tr_id}', tag=TR_NAME_INPUT)

def get_req_tr_name_value(image: Image) -> str:
    base_name = dpg.get_value(TR_NAME_INPUT)
    if not base_name:
        raise ValueError('A name for the new image must be provided')
    return dpg.get_value(TR_NAME_INPUT) + image.format.to_extension()

def build_op_img_selector(image_name: str) -> None:
    image_list = list(map(lambda img: img.name, img_repo.get_same_shape_images(image_name)))
    dpg.add_text('Select Another Image to combine')
    dpg.add_listbox(image_list, tag=TR_IMG_INPUT)

def get_req_tr_img_value() -> Image:
    img_name = dpg.get_value(TR_IMG_INPUT)
    if not img_repo.contains_image(img_name):
        raise ValueError('Selecting a valid image is required for transformation')
    return img_repo.get_image(img_name)

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
    return Image(new_name, image.format, image.data)


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
    if image.shape != sec_image.shape:
        raise ValueError('You can only sum images with the same shape')

    # 2. Procesamos
    new_data = sum_images(image, sec_image)
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
    if image.shape != sec_image.shape:
        raise ValueError('You can only sub images with the same shape')

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
    if image.shape != sec_image.shape:
        raise ValueError('You can only multiply images with the same shape')

    # 2. Procesamos
    new_data = multiply_images(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return Image(new_name, image.format, new_data)


TRANSFORMATIONS: Dict[str, Callable[[str], None]] = {
    TR_NOP: build_nop_dialog,
    TR_ADD: build_add_dialog,
    TR_SUB: build_sub_dialog,
    TR_MULT: build_mult_dialog,
}
