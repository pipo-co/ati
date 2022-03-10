from typing import Dict, Callable

import dearpygui.dearpygui as dpg

import images_repo as img_repo
import interface
from image_utils import Image, strip_extension, sum_images
from interface_utils import render_error

def finalize_transformation(new_image: Image) -> None:
    img_repo.persist_image(new_image)
    interface.register_image(new_image)
    interface.render_image_window(new_image.name)

def build_tr_dialog(tr_id: str) -> int:
    # TODO: Estaria bueno que fuese un modal (modal=True), pero no me esta andando.
    #  No deberias poder interactuar con nada hasta resolver el dialogo
    return dpg.window(label=f'Apply {tr_id.capitalize()} Transformation', no_move=True, no_close=True, no_collapse=True, pos=(800, 512))

def build_tr_name_input(tr_id: str, image_name: str) -> None:
    dpg.add_text('Select New Image Name (no extension)')
    dpg.add_input_text(default_value=strip_extension(image_name) + f'_{tr_id}', tag=f'tr_{tr_id}_name_{image_name}')

def get_tr_name_value(tr_id: str, image: Image) -> str:
    return dpg.get_value(f'tr_{tr_id}_name_{image.name}') + image.format.to_extension()

def build_op_img_selector(tr_id: str, image_name: str) -> None:
    image_list = list(map(lambda img: img.name, img_repo.get_compatible_images(image_name)))
    dpg.add_text('Select Another Image to combine')
    dpg.add_listbox(image_list, tag=f'tr_{tr_id}_img_{image_name}')

def get_tr_img_value(tr_id: str, image: Image) -> str:
    return dpg.get_value(f'tr_{tr_id}_img_{image.name}')

def build_tr_dialog_end_buttons(image_name: str, dialog: int, handle: Callable[[str, int], None]) -> None:
    with dpg.group(horizontal=True):
        dpg.add_button(label='Transform', user_data=(image_name, dialog), callback=lambda s, ap, ud: handle(ud[0], ud[1]))
        dpg.add_button(label='Cancel', user_data=dialog, callback=lambda s, ad, ud: dpg.delete_item(ud))


NOP_TRANSFORMATION: str = 'nop'
@render_error
def build_nop_dialog(image_name: str) -> None:
    with build_tr_dialog(NOP_TRANSFORMATION) as dialog:
        build_tr_name_input(NOP_TRANSFORMATION, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_dialog_end_buttons(image_name, dialog, nop_handle)

def nop_handle(image_name: str, dialog) -> None:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_tr_name_value(NOP_TRANSFORMATION, image)
    # 2. Eliminamos modal
    dpg.delete_item(dialog)
    # 3. Procesamos - Deberia ser async
    # Do Nothing
    # 4. Creamos Imagen y finalizamos
    new_image = Image(new_name, image.format, image.data)
    finalize_transformation(new_image)


ADD_OPERATION: str = 'add'
@render_error
def build_add_dialog(image_name: str) -> None:
    with build_tr_dialog(ADD_OPERATION) as dialog:
        build_tr_name_input(ADD_OPERATION, image_name)
        build_op_img_selector(ADD_OPERATION, image_name)
        build_tr_dialog_end_buttons(image_name, dialog, sum_handle)

def sum_handle(image_name: str, dialog) -> None:
    # 1. Obtenemos inputs
    image = img_repo.get_image(image_name)
    new_name: str = get_tr_name_value(ADD_OPERATION, image)
    sec_image = img_repo.get_image(get_tr_img_value(ADD_OPERATION, image))
    # 2. Eliminamos modal
    dpg.delete_item(dialog)
    # 3. Procesamos - Deberia ser async
    new_data = sum_images(image, sec_image)
    # 4. Creamos Imagen y finalizamos
    new_image = Image(new_name, image.format, new_data)
    finalize_transformation(new_image)


TRANSFORMATIONS: Dict[str, Callable[[str], None]] = {
    NOP_TRANSFORMATION: build_nop_dialog,
    ADD_OPERATION: build_add_dialog,
}
