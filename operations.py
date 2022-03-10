from typing import Dict, Callable

import dearpygui.dearpygui as dpg

import images_repo as img_repo
import interface
from image_utils import Image, strip_extension, sum_images
from interface_utils import render_error

second_img_name: str = ' '

def finalize_transformation(new_image: Image) -> None:
    img_repo.persist_image(new_image)
    interface.register_image(new_image)
    interface.render_image_window(new_image.name)

def build_op_dialog(op_id: str) -> int:
    # TODO: Estaria bueno que fuese un modal (modal=True), pero no me esta andando.
    #  No deberias poder interactuar con nada hasta resolver el dialogo
    return dpg.window(label=f'Apply {op_id.capitalize()} Operation', no_move=True, no_close=True, no_collapse=True, pos=(800, 512))

def build_op_name_input(op_id: str, image_name: str) -> None:
    dpg.add_text('Select New Image Name (no extension)')
    dpg.add_input_text(default_value=strip_extension(image_name) + f'_{op_id}', tag=f'op_{op_id}_name_{image_name}')

def build_op_dialog_end_buttons(image_name: str, sec_img_name: str, dialog: int, handle: Callable[[str, int], None]) -> None:
    with dpg.group(horizontal=True):
        dpg.add_button(label='Operate', user_data={"img_name": image_name, "sec_img_name": sec_img_name, "dialog": dialog}, callback=lambda s, ap, ud: handle(ud['img_name'], ud['sec_img_name'], ud['dialog']))
        dpg.add_button(label='Cancel', user_data=dialog, callback=lambda s, ad, ud: dpg.delete_item(ud))

def build_op_img_selector(image_name: str) -> str:
    dpg.add_text('Select Second Image to combine')
    with dpg.menu(label="Compatible Images", tag='menu') as menu:
        for img in img_repo.get_compatible_images(image_name):
            dpg.add_button(label=img.name.capitalize(), user_data=img.name, callback=lambda s, ad, ud: dpg.set_item_user_data(menu, ud))

ADD_OPERATION: str = 'add'
@render_error
def build_add_dialog(image_name: str) -> None:
    with build_op_dialog(ADD_OPERATION) as dialog:
        build_op_name_input(ADD_OPERATION, image_name)
        build_op_img_selector(image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.

        build_op_dialog_end_buttons(image_name, second_img_name, dialog, sum_handle)

def sum_handle(image_name: str, second_img_name: str, dialog) -> None:
    # 1. Obtenemos inputs
    print(second_img_name)
    image = img_repo.get_image(image_name)
    sec_image = img_repo.get_image(dpg.get_item_user_data('menu'))
    new_name: str = dpg.get_value(f'op_add_name_{image_name}') + image.format.to_extension()
    # 2. Eliminamos modal
    dpg.delete_item(dialog)
    second_img_name = " "
    # 3. Procesamos - Deberia ser async
    new_data = sum_images(image, sec_image)
    # 4. Creamos Imagen y finalizamos
    new_image = Image(new_name, image.format, new_data)
    finalize_transformation(new_image)


OPERATIONS: Dict[str, Callable[[str], None]] = {
    ADD_OPERATION: build_add_dialog,
}
