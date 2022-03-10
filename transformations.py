from typing import Dict, Callable

import dearpygui.dearpygui as dpg

import images_repo as img_repo
import interface
from image_utils import Image, strip_extension, sum_images
from interface_utils import render_error

# General Items
TR_DIALOG: str = 'tr_dialog'

# Custom Inputs
TR_NAME_INPUT: str = 'tr_name_input'
TR_IMG_INPUT: str = 'tr_img_input'

def finalize_transformation(new_image: Image) -> None:
    img_repo.persist_image(new_image)
    interface.register_image(new_image)
    interface.render_image_window(new_image.name)

def build_tr_dialog(tr_id: str) -> int:
    return dpg.window(label=f'Apply {tr_id.capitalize()} Transformation', tag=TR_DIALOG, modal=True, no_close=True, pos=interface.CENTER_POS)

@render_error
def delete_dialog() -> None:
    dpg.delete_item(TR_DIALOG)

def build_tr_name_input(tr_id: str, image_name: str) -> None:
    dpg.add_text('Select New Image Name (no extension)')
    dpg.add_input_text(default_value=strip_extension(image_name) + f'_{tr_id}', tag=TR_NAME_INPUT)

def get_tr_name_value(image: Image) -> str:
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

def build_tr_dialog_end_buttons(tr_id: str, image_name: str, handle: Callable[[str], None]) -> None:
    with dpg.group(horizontal=True):
        dpg.add_button(label='Transform', user_data=(handle, image_name), callback=lambda s, ap, ud: ud[0](ud[1]))
        dpg.add_button(label='Cancel', user_data=tr_id, callback=lambda: delete_dialog())


TR_NOP: str = 'nop'
@render_error
def build_nop_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOP):
        build_tr_name_input(TR_NOP, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_dialog_end_buttons(TR_NOP, image_name, nop_handle)

@render_error
def nop_handle(image_name: str) -> None:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_tr_name_value(image)
    # 2. Eliminamos modal
    delete_dialog()
    # 3. Procesamos - Deberia ser async
    # Do Nothing
    # 4. Creamos Imagen y finalizamos
    new_image = Image(new_name, image.format, image.data)
    finalize_transformation(new_image)


TR_ADD: str = 'add'
@render_error
def build_add_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_ADD):
        build_tr_name_input(TR_ADD, image_name)
        build_op_img_selector(image_name)
        build_tr_dialog_end_buttons(TR_ADD, image_name, sum_handle)

@render_error
def sum_handle(image_name: str) -> None:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_tr_name_value(image)
    sec_image = get_req_tr_img_value()
    if image.shape != sec_image.shape:
        raise ValueError('You can only sum images with the same shape')

    # 2. Eliminamos modal
    delete_dialog()
    # 3. Procesamos - Deberia ser async
    new_data = sum_images(image, sec_image)
    # 4. Creamos Imagen y finalizamos
    new_image = Image(new_name, image.format, new_data)
    finalize_transformation(new_image)


TRANSFORMATIONS: Dict[str, Callable[[str], None]] = {
    TR_NOP: build_nop_dialog,
    TR_ADD: build_add_dialog,
}
