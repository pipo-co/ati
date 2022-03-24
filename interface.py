from typing import Callable, Tuple, Union

import dearpygui.dearpygui as dpg
import numpy as np

from image_utils import image_to_rgba_array, load_image, valid_image_formats, Image, save_image, get_extension, \
    create_square_image, create_circle_image, CIRCLE_IMAGE_NAME, SQUARE_IMAGE_NAME
import images_repo as img_repo
from interface_utils import render_error
from metadata_repo import set_metadata_file
from transformations import build_transformations_menu

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
MIN_IMAGE_WIDTH: int = 235
MIN_IMAGE_HEIGHT: int = 200
HIST_OFFSET: int = 20
HIST_WIDTH: int = 200

# Creates window only if it doesn't exist
@render_error
def render_image_window(image_name: str):
    if dpg.does_item_exist(f'image_{image_name}'):
        dpg.focus_item(f'image_{image_name}')
    else:
        # Precalculamos las operaciones costosas.
        # Si lo hacemos durante el rendereado, se generan condiciones de carrera.
        image: Image = img_repo.get_image(image_name)
        width, height = calculate_image_window_size(image)
        hists = image.get_histograms()

        with dpg.window(label=image_name, tag=f'image_window_{image_name}', width=width, height=height, no_scrollbar=True, no_resize=True, user_data={'image_name': image_name, 'hists_toggled': False}, on_close=lambda: dpg.delete_item(window)) as window:
            with dpg.menu_bar():
                dpg.add_menu_item(label='Save', user_data=image_name, callback=lambda s, ad, ud: trigger_save_image_dialog(ud))
                build_transformations_menu(image_name)
                dpg.add_menu_item(label='Show Histograms', tag=f'hists_toggle_{image_name}', user_data=image_name, callback=lambda s, ad, ud: toggle_hists(ud))

            with dpg.group(horizontal=True):
                with dpg.group(width=image.width):
                    dpg.add_image(image_name, tag=f'image_{image_name}', width=image.width, height=image.height)
                    with dpg.group(horizontal=True):
                        dpg.add_text(f'Height {image.height}')
                        dpg.add_separator()
                        dpg.add_text(f'Width {image.width}')
                        dpg.add_separator()
                        dpg.add_text(f'Type {image.type}')
                    dpg.add_text('', tag=f'image_{image_name}_region')
                    dpg.add_text('', tag=f'image_{image_name}_pointer')

                with dpg.group(tag=f'hist_group_{image.name}', width=HIST_WIDTH, show=False):
                    if len(hists) == 1:
                        build_reduced_histogram_plot(image_name, 'grey_hist_theme',  *hists[0])
                    else:
                        build_reduced_histogram_plot(image_name, 'red_hist_theme',   *hists[Image.RED_CHANNEL])
                        build_reduced_histogram_plot(image_name, 'green_hist_theme', *hists[Image.GREEN_CHANNEL])
                        build_reduced_histogram_plot(image_name, 'blue_hist_theme',  *hists[Image.BLUE_CHANNEL])

def calculate_image_window_size(image: Image) -> Tuple[int, int]:
    # 15 = padding, 120 = menu_bar + info_size
    return max(image.width, MIN_IMAGE_WIDTH) + 15, max(image.height, MIN_IMAGE_HEIGHT) + 120

@render_error
def toggle_hists(image_name: str) -> None:
    window = f'image_window_{image_name}'
    toggle = f'hists_toggle_{image_name}'
    hists = f'hist_group_{image_name}'
    user_data = dpg.get_item_user_data(window)
    plots_toggled: bool = user_data['hists_toggled']
    diff: int = HIST_WIDTH + 10
    if plots_toggled:
        diff = -diff
        dpg.set_item_label(toggle, 'Show Histograms')
        dpg.hide_item(hists)
    else:
        dpg.set_item_label(toggle, 'Hide Histograms')
        dpg.show_item(hists)
    dpg.set_item_width(window, dpg.get_item_width(window) + diff)
    user_data['hists_toggled'] = not plots_toggled

def build_hist_themes():
    with dpg.theme(tag='red_hist_theme'):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(value=(255, 0, 0), category=dpg.mvThemeCat_Plots)
    with dpg.theme(tag='green_hist_theme'):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(value=(0, 255, 0), category=dpg.mvThemeCat_Plots)
    with dpg.theme(tag='blue_hist_theme'):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(value=(0, 0, 255), category=dpg.mvThemeCat_Plots) 
    with dpg.theme(tag='grey_hist_theme'):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(value=(127, 127, 127), category=dpg.mvThemeCat_Plots)

def build_histogram_plot(theme: str, hist: np.ndarray, bins: np.ndarray, height: int = 0, no_ticks: bool = False, no_mouse_pos: bool = False) -> None:
    with dpg.plot(height=height, no_mouse_pos=no_mouse_pos):
        dpg.add_plot_axis(dpg.mvXAxis, no_tick_marks=no_ticks, no_tick_labels=no_ticks)
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, no_tick_marks=no_ticks, no_tick_labels=no_ticks)
        grey_hist = dpg.add_bar_series(bins, hist, parent=y_axis) # noqa
        dpg.bind_item_theme(grey_hist, theme)

def build_reduced_histogram_plot(image_name: str, theme: str, hist: np.ndarray, bins: np.ndarray):
    build_histogram_plot(theme, hist, bins, height=HIST_WIDTH * 9 // 16, no_ticks=True, no_mouse_pos=True)
    dpg.add_button(label='Expand', user_data=(image_name, theme, hist, bins), callback=lambda s, ad, ud: build_expanded_histogram_plot(*ud))
    dpg.add_separator()

@render_error
def build_expanded_histogram_plot(image_name: str, theme: str, hist: np.ndarray, bins: np.ndarray) -> None:
    with dpg.window(label=f'Histogram - {image_name} - {theme.split("_")[0].capitalize()} Channel'):
        build_histogram_plot(theme, hist.tolist(), bins.tolist()) # noqa - Hackazo para evitar el bug de que se renderea mal

def register_image(image: Image) -> None:
    image_vector = image_to_rgba_array(image)
    dpg.add_static_texture(image.width, image.height, image_vector, tag=image.name, parent=TEXTURE_REGISTRY) # noqa
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
                dpg.set_value(pointer, f"Pixel: {get_pixel_pos_in_image(window, image_name)}  Value: {image.get_pixel(pixel)}")
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
