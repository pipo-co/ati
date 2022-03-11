from typing import Callable, Tuple, Union

import dearpygui.dearpygui as dpg
import numpy as np

from image_utils import image_to_rgba_array, load_image, valid_image_formats, Image, save_image, get_extension, \
    create_square_image, create_circle_image, CIRCLE_IMAGE_NAME, SQUARE_IMAGE_NAME
import images_repo as img_repo
from interface_utils import render_error
from metadata_repo import set_metadata_file
from transformations import TRANSFORMATIONS

# General Items
PRIMARY_WINDOW: str = 'primary'
IMAGES_MENU: str = 'images_menu'

# Registries
TEXTURE_REGISTRY: str = 'texture_registry'

# Dialog Tags
LOAD_IMAGE_DIALOG: str = 'load_image_dialog'
SAVE_IMAGE_DIALOG: str = 'save_image_dialog'
LOAD_METADATA_DIALOG: str = 'load_metadata_dialog'

CENTER_POS: Tuple[int, int] = (750, 350)
MIN_IMAGE_SIZE: Tuple[int, int] = (200, 200)

# Creates window only if it doesn't exist
@render_error
def render_image_window(image_name: str):
    if dpg.does_item_exist(f'image_{image_name}'):
        dpg.focus_item(f'image_{image_name}')
    else:
        with dpg.window(label=image_name, tag=f'image_window_{image_name}', pos=CENTER_POS, min_size=MIN_IMAGE_SIZE, no_resize=True, user_data={'image_name': image_name}, on_close=lambda: dpg.delete_item(window)) as window:
            with dpg.menu_bar():
                dpg.add_menu_item(label='Save', user_data=image_name, callback=lambda s, ad, ud: trigger_save_image_dialog(ud))
                with dpg.menu(label='Transform'):
                    for name, tr in TRANSFORMATIONS.items():
                        dpg.add_menu_item(label=name.capitalize(), user_data=(tr, image_name), callback=lambda s, ad, ud: ud[0](ud[1]))

            image: Image = img_repo.get_image(image_name)
            dpg.add_image(image_name, tag=f'image_{image_name}')
            dpg.add_text(f'Height: {image.height}  Width: {image.width}')
            dpg.add_text('', tag=f'image_{image_name}_region')
            dpg.add_text('', tag=f'image_{image_name}_pointer')

            
def register_image(image: Image) -> None:
    image_vector = image_to_rgba_array(image)
    dpg.add_static_texture(image.width, image.height, image_vector, tag=image.name, parent=TEXTURE_REGISTRY)
    dpg.add_menu_item(label=image.name, parent=IMAGES_MENU, user_data=image.name, callback=lambda s, ad, ud: render_image_window(ud))

@render_error
def load_image_handler(app_data):
    path = app_data['file_path_name']
    image_name = Image.name_from_path(path)

    if not img_repo.contains_image(image_name):
        image = load_image(path)
        img_repo.persist_image(image)
        register_image(image)

    render_image_window(image_name)

@render_error
def save_image_handler(app_data, image_name: str) -> None:
    image = img_repo.get_image(image_name)
    dir_path = app_data['file_path_name']
    save_image(image, dir_path)

def load_metadata_handler(app_data):
    path = app_data['file_path_name']
    if not get_extension(path) == '.tsv':
        raise ValueError('Metadata file must be a tsv')
    set_metadata_file(path)

# Generic function to create images
def create_image(name: str, supplier: Callable[[], Image]) -> None:
    if not img_repo.contains_image(name):
        image = supplier()
        img_repo.persist_image(image)
        register_image(image)

    render_image_window(name)

def create_circle_handler():
    create_image(CIRCLE_IMAGE_NAME, create_circle_image)

def create_square_handler():
    create_image(SQUARE_IMAGE_NAME, create_square_image)

@render_error
def trigger_save_image_dialog(image_name: str) -> None:
    dpg.set_item_label(SAVE_IMAGE_DIALOG, f'Choose where to save {image_name}...')
    dpg.set_item_user_data(SAVE_IMAGE_DIALOG, image_name)
    dpg.show_item(SAVE_IMAGE_DIALOG)

def build_save_image_dialog() -> None:
    dpg.add_file_dialog(tag=SAVE_IMAGE_DIALOG, default_path='images', directory_selector=True, show=False, modal=True, width=1024, height=512, callback=lambda s, ad, ud: save_image_handler(ad, ud))

def build_load_image_dialog() -> None:
    with dpg.file_dialog(label='Choose file to load...', tag=LOAD_IMAGE_DIALOG, default_path='images', directory_selector=False, show=False, modal=True, width=1024, height=512, callback=lambda s, ad: load_image_handler(ad)):
        dpg.add_file_extension(f'Image{{{",".join(valid_image_formats())}}}')

def build_load_metadata_dialog() -> None:
    with dpg.file_dialog(label='Choose metadata file to load...', tag=LOAD_METADATA_DIALOG, default_path='images', directory_selector=False, show=False, modal=True, width=1024, height=512, callback=lambda s, ad: load_metadata_handler(ad)):
        dpg.add_file_extension('.tsv')

def get_pixel_pos_in_image(window: Union[int, str], image_name: str) -> Tuple[int, int]:
    mouse_pos = dpg.get_mouse_pos(local=False)
    window_pos = dpg.get_item_pos(window)
    img_pos = dpg.get_item_pos(f'image_{image_name}')
    return int(mouse_pos[0] - window_pos[0] - img_pos[0]), int(mouse_pos[1] - window_pos[1] - img_pos[1])

def is_image_window(window: Union[int, str]) -> bool:
    return isinstance(window, str) and window.startswith('image_window')

def get_focused_hovered_image_window() -> Union[int, str]:
    window = dpg.get_active_window()
    if is_image_window(window) and dpg.is_item_hovered(window):
        return window
    else:
        return 0

def build_image_handler_registry() -> None:
    @render_error
    def mouse_move_handler() -> None:
        window = get_focused_hovered_image_window()

        # Cleanup de las otras image windows
        for win_id in dpg.get_windows():
            win = dpg.get_item_alias(win_id)
            if window != win and is_image_window(win):
                user_data = dpg.get_item_user_data(win)
                win_img_name = user_data['image_name']
                pointer = f'image_{win_img_name}_pointer'
                selection = f'image_{win_img_name}_selection'

                dpg.set_value(pointer, '')
                user_data.pop('init_draw', None)
                if dpg.does_item_exist(selection):
                    dpg.delete_item(selection)

        if window == 0:
            return  # No hay ninguna imagen seleccionada

        user_data = dpg.get_item_user_data(window)
        image_name: str = user_data['image_name']
        pointer = f'image_{image_name}_pointer'
        selection = f'image_{image_name}_selection'

        if dpg.is_item_focused(window):
            image = img_repo.get_image(image_name)
            pixel = get_pixel_pos_in_image(window, image_name)

            if image.valid_pixel(pixel):
                # Estamos en la imagen! -> Dibujamos
                usr_data = dpg.get_item_user_data(window)
                if 'init_draw' in usr_data:
                    if dpg.does_item_exist(selection):
                        dpg.delete_item(selection)

                    dpg.draw_rectangle(usr_data['init_draw'], pixel, parent=window, tag=selection, color=(0xCC, 0x00, 0x66, 200))

                dpg.show_item(pointer)
                dpg.set_value(pointer, f"Pixel: {get_pixel_pos_in_image(window, image_name)}  Value: {image.data[int(pixel[1])][int(pixel[0])]}")
                return  # Terminamos de dibujar

        # No estamos en la imagen -> Borramos lo dibujado
        user_data.pop('init_draw', None)
        dpg.set_value(pointer, '')
        if dpg.does_item_exist(selection):
            dpg.delete_item(selection)

    @render_error
    def mouse_down_handler():
        window = get_focused_hovered_image_window()

        # Cleanup de las otras image windows
        for win_id in dpg.get_windows():
            win = dpg.get_item_alias(win_id)
            if window != win and is_image_window(win):
                user_data = dpg.get_item_user_data(win)
                win_img_name = user_data['image_name']
                region = f'image_{win_img_name}_region'
                selection = f'image_{win_img_name}_selection'

                dpg.set_value(region, '')
                if dpg.does_item_exist(selection):
                    dpg.delete_item(selection)

        if window == 0:
            return  # No hay ninguna imagen seleccionada

        usr_data = dpg.get_item_user_data(window)
        image_name: str = usr_data['image_name']
        usr_data['init_draw'] = usr_data.get('init_draw', get_pixel_pos_in_image(window, image_name))
        dpg.set_item_user_data(window, usr_data)

    @render_error
    def mouse_release_handler():
        window = get_focused_hovered_image_window()

        if window == 0:
            return  # No hay ninguna imagen seleccionada

        usr_data = dpg.get_item_user_data(window)
        image_name: str = usr_data['image_name']
        image = img_repo.get_image(image_name)
        region = f'image_{image_name}_region'
        selection = f'image_{image_name}_selection'

        if 'init_draw' not in usr_data:
            return  # No se detecto el inicio del click. Probablemente se este arrastrando la ventana.

        pixel_pos = get_pixel_pos_in_image(window, image_name)
        init_draw = usr_data['init_draw']
        region_size = (pixel_pos[0] - init_draw[0] + 1, pixel_pos[1] - init_draw[1] + 1)

        x = (int(min(pixel_pos[0], init_draw[0])), int(max(pixel_pos[0], init_draw[0])))
        y = (int(min(pixel_pos[1], init_draw[1])), int(max(pixel_pos[1], init_draw[1])))

        if x[1] > x[0] and y[1] > y[0]:
            if image.channels > 1:
                # TODO: A veces lanza una excepcion: 'RuntimeWarning: Mean of empty slice.'
                mean = np.mean(np.array(image.data[y[0]:y[1], x[0]:x[1]]).reshape((-1, 3)), axis=0)
            else:
                mean = np.mean(image.data[y[0]:y[1], x[0]:x[1]])

            dpg.set_value(region, f"#Pixel: {np.prod(np.abs(region_size))}  Avg: {np.around(mean, 2)}")
            dpg.show_item(region)
        else:
            dpg.set_value(region, '')
            if dpg.does_item_exist(selection):
                dpg.delete_item(selection)

        usr_data.pop('init_draw', None)

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=mouse_move_handler)
        dpg.add_mouse_down_handler(callback=mouse_down_handler)
        dpg.add_mouse_release_handler(callback=mouse_release_handler)
