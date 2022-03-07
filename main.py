import os
from typing import Dict

import dearpygui.dearpygui as dpg
import numpy as np

from image_utils import image_to_rgba_array, open_image

loaded_images: Dict[str, np.ndarray] = {}

def print_me(sender):
    print(f"Menu Item: {sender}")

# Creates window only if it doesn't exist
def render_image_window(image_tag: str):
    if dpg.does_alias_exist(f'image_{image_tag}'):
        print('focus')
        dpg.focus_item(f'image_{image_tag}')
    else:
        print('render')
        with dpg.window(label=os.path.basename(image_tag), tag=f'window_{image_tag}', no_resize=True, on_close=lambda: dpg.delete_item(f'window_{image_tag}')):
            dpg.add_image(image_tag, tag=f'image_{image_tag}')

def load_image(sender, app_data):
    path = app_data['file_path_name']

    if path not in loaded_images:
        image = open_image(path)
        loaded_images[path] = image

        image_vector = image_to_rgba_array(image)
        image_shape = image.shape
        width = image_shape[0]
        height = image_shape[1]
        dpg.add_static_texture(width, height, image_vector, tag=path, parent='texture_registry', user_data=path)
        dpg.add_image_button(path, label=path, parent='images_menu', tag=f'button_{path}', user_data=path, width=128, height=128, callback=lambda s, ad, ud: render_image_window(ud))

    render_image_window(path)


def main():
    dpg.create_context()
    dpg.create_viewport(title='Analisis y Tratamiento de Imagenes')  # TODO: Add icons
    dpg.setup_dearpygui()

    # Image Registry
    dpg.add_texture_registry(tag='texture_registry')

    # Handler Registries
    with dpg.item_handler_registry(tag='texture_registry_handler'):
        dpg.add_item_activated_handler(callback=lambda sender, app_data, user_data: render_image_window(user_data))

    # File Dialog
    with dpg.file_dialog(label='Choose file to load...', tag='file_dialog', default_path='images', directory_selector=False, show=False, modal=True, width=1024, height=512, callback=load_image):
        dpg.add_file_extension("Image{.pgm,.PGM,.ppm,.PPM,.raw,.RAW}")

    with dpg.viewport_menu_bar(tag='vp_menu_bar', label="Example Window"):
        dpg.add_menu_item(label="Load New Image", callback=lambda: dpg.show_item('file_dialog'))

        dpg.add_menu(label="Images", tag='images_menu')

        with dpg.menu(label="Configuration"):
            dpg.add_text('Metadata file for raw images: ')
            dpg.add_input_text(default_value='images/raw_metadata.csv')

    with dpg.window(tag='primary', label="Example Window"):
        with dpg.window(tag='another', label="Example Window"):
            dpg.add_text("Hello, world")
            dpg.add_button(label="Save")
            dpg.add_input_text(label="string", default_value="Quick brown fox")
            dpg.add_slider_float(label="float", default_value=0.273, max_value=1)

    dpg.show_viewport()
    dpg.maximize_viewport()  # Max viewport size
    dpg.set_primary_window('primary', True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
