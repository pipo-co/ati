import dearpygui.dearpygui as dpg

from image_utils import image_to_rgba_array, load_image, load_metadata, valid_image_formats, Image, save_image, get_extension
import images_repo as img_repo
from interface_utils import render_error
from metadata_repo import load_metadata
from transformations import TRANSFORMATIONS

# General Items
PRIMARY_WINDOW: str = 'primary'
IMAGES_MENU: str = 'images_menu'

# Registries
TEXTURE_REGISTRY: str = 'texture_registry'

# Dialog Tags
LOAD_IMAGE_DIALOG: str = 'load_image_dialog'
SAVE_IMAGE_DIALOG: str = 'save_image_dialog'

SAVE_METADATA_DIALOG: str = 'save_metadata_dialog'

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
                with dpg.menu(label="Apply Transformation"):
                    for name, tr in TRANSFORMATIONS.items():
                        dpg.add_button(label=name.capitalize(), user_data=image_name, callback=lambda s, ad, ud: tr(ud))

def register_image(image: Image) -> None:
    image_vector = image_to_rgba_array(image)
    dpg.add_static_texture(image.width, image.height, image_vector, tag=image.name, parent=TEXTURE_REGISTRY)
    dpg.add_button(label=image.name, parent=IMAGES_MENU, tag=f'button_{image.name}', user_data=image.name, callback=lambda s, ad, ud: render_image_window(ud))

@render_error
def load_image_handler(app_data):
    path = app_data['file_path_name']
    image_name = Image.name_from_path(path)

    if not img_repo.contains_image(image_name):
        image = load_image(path)
        img_repo.persist_image(image)
        register_image(image)

    render_image_window(image_name)

def save_image_handler(app_data, image_name: str) -> None:
    image = img_repo.get_image(image_name)
    dir_path = app_data['file_path_name']
    save_image(image, dir_path)
    dpg.delete_item(f'{SAVE_IMAGE_DIALOG}_{image_name}')

def load_metadata_handler(app_data):
    path = app_data['file_path_name']
    if get_extension(path) == '.csv':
        load_metadata(path)


# TODO: eliminar item tambien cuando se cancela
def build_save_image_dialog(image_name: str) -> None:
    dpg.add_file_dialog(label=f'Choose where to save {image_name}...', tag=f'{SAVE_IMAGE_DIALOG}_{image_name}', default_path='images', directory_selector=True, modal=True, width=1024, height=512, user_data=image_name, callback=lambda s, ad, ud: save_image_handler(ad, ud))

def build_load_image_dialog() -> None:
    with dpg.file_dialog(label='Choose file to load...', tag=LOAD_IMAGE_DIALOG, default_path='images', directory_selector=False, show=False, modal=True, width=1024, height=512, callback=lambda s, ad: load_image_handler(ad)):
        dpg.add_file_extension(f'Image{{{",".join(valid_image_formats())}}}')

def build_load_metadata_dialog() -> None:
    with dpg.file_dialog(label='Choose file to load...', tag=SAVE_METADATA_DIALOG, default_path='images', directory_selector=False, show=False, modal=True, width=1024, height=512, callback=lambda s, ad: load_metadata_handler(ad)):
        dpg.add_file_extension('.csv')