import functools
from typing import Dict

import dearpygui.dearpygui as dpg

from image_utils import image_to_rgba_array, load_image, valid_image_formats, Image, save_image

# Global Data
raw_images_metadata_path: str = 'images/raw_metadata.csv'
loaded_images: Dict[str, Image] = {}  # Images by name

# General Tags
PRIMARY_WINDOW: str = 'primary'

# Registry Tags

# Dialog Tags
LOAD_IMAGE_DIALOG: str = 'load_image_dialog'
SAVE_IMAGE_DIALOG: str = 'save_image_dialog'

# Renderea errores en popup
def render_error(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            build_error_popup_from_error(e)
            raise e
    return decorator

def build_error_popup(error_msg: str, popup_tag: int = 0) -> None:
    with dpg.window(label='Error', no_move=True, no_resize=True, no_title_bar=True, pos=(0, 19), height=5, tag=popup_tag) as popup:
        dpg.add_text(f'An Error Occurred: {error_msg}')
        dpg.add_button(label='Close', width=50, height=30, callback=lambda: dpg.delete_item(popup))

def build_error_popup_from_error(e: Exception) -> None:
    popup_tag = id(e)
    if not dpg.does_item_exist(popup_tag):
        build_error_popup(str(e), popup_tag)

# Creates window only if it doesn't exist
@render_error
def render_image_window(image_name: str):
    if dpg.does_item_exist(f'image_{image_name}'):
        dpg.focus_item(f'image_{image_name}')
    else:
        with dpg.window(label=image_name, tag=f'window_{image_name}', pos=(100, 100), no_resize=True, on_close=lambda: dpg.delete_item(f'window_{image_name}')):
            dpg.add_image(image_name, tag=f'image_{image_name}')
            with dpg.menu_bar():
                dpg.add_menu_item(label="Save Image", user_data=image_name, callback=lambda s, ad, ud: build_save_image_dialog(ud))
                # with dpg.menu(label="Apply Transformation"):

@render_error
def load_image_handler(app_data):
    path = app_data['file_path_name']
    image_name = Image.name_from_path(path)

    if image_name not in loaded_images:
        image = load_image(path)
        loaded_images[image_name] = image

        image_vector = image_to_rgba_array(image)
        dpg.add_static_texture(image.width, image.height, image_vector, tag=image_name, parent='texture_registry')
        dpg.add_button(label=image_name, parent='images_menu', tag=f'button_{image_name}', user_data=image_name, callback=lambda s, ad, ud: render_image_window(ud))

    render_image_window(image_name)

def save_image_handler(app_data, image_name: str) -> None:
    image = loaded_images[image_name]
    dir_path = app_data['file_path_name']
    save_image(image, dir_path)
    dpg.delete_item(f'{SAVE_IMAGE_DIALOG}_{image_name}')

def build_save_image_dialog(image_name: str) -> None:
    dpg.add_file_dialog(label=f'Choose where to save {image_name}...', tag=f'{SAVE_IMAGE_DIALOG}_{image_name}', default_path='images', directory_selector=True, modal=True, width=1024, height=512, user_data=image_name, callback=lambda s, ad, ud: save_image_handler(ad, ud))

def build_load_image_dialog() -> None:
    with dpg.file_dialog(label='Choose file to load...', tag=LOAD_IMAGE_DIALOG, default_path='images', directory_selector=False, show=False, modal=True, width=1024, height=512, callback=lambda s, ad: load_image_handler(ad)):
        dpg.add_file_extension(f'Image{{{",".join(valid_image_formats())}}}')

def main():
    dpg.create_context()
    dpg.create_viewport(title='Analisis y Tratamiento de Imagenes')  # TODO: Add icons
    dpg.setup_dearpygui()

    # Image Registry
    dpg.add_texture_registry(tag='texture_registry')

    # File Dialog
    build_load_image_dialog()

    with dpg.viewport_menu_bar(tag='vp_menu_bar', label="Example Window"):
        dpg.add_menu_item(label="Load New Image", callback=lambda: dpg.show_item(LOAD_IMAGE_DIALOG))

        dpg.add_menu(label="Images", tag='images_menu')

        with dpg.menu(label="Configuration"):
            dpg.add_text('Metadata file for raw images: ')
            dpg.add_input_text(default_value=raw_images_metadata_path)

    dpg.add_window(tag=PRIMARY_WINDOW)

    dpg.show_viewport()
    dpg.maximize_viewport()  # Max viewport size
    dpg.set_primary_window(PRIMARY_WINDOW, True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
