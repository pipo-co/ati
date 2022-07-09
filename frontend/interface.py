import os
from typing import Callable, Tuple, Union, Dict, Optional, List

import dearpygui.dearpygui as dpg
import numpy as np
from models.draw_cmd import CircleDrawCmd, LineDrawCmd, ScatterDrawCmd

from models.image import image_to_rgba_array, load_image, valid_image_formats, Image, save_image, get_extension, \
    create_square_image, create_circle_image, CIRCLE_IMAGE_NAME, SQUARE_IMAGE_NAME
from models.movie import Movie
from models.path_utils import movie_dir_selections

from repositories import images_repo as img_repo, movies_repo as mov_repo
from .interface_utils import render_error
from repositories.metadata_repo import set_metadata_file
from .transformations import build_transformations_menu

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
HISTORY_WIDTH: int = 400

# Creates window only if it doesn't exist
@render_error
def render_image_window(image_name: str, movie: Optional[Movie] = None, pos: Union[List[int], Tuple[int, ...]] = ()):
    if dpg.does_item_exist(f'image_{image_name}'):
        dpg.focus_item(f'image_{image_name}')
    else:
        # Precalculamos las operaciones costosas.
        # Si lo hacemos durante el rendereado, se generan condiciones de carrera.
        image: Image = img_repo.get_image(image_name)
        width, height = calculate_image_window_size(image)
        hists = image.get_histograms()

        window_label = f'Movie: {movie.name} - Frame {movie.current_frame}' if movie else image_name
        with dpg.window(label=window_label, tag=f'image_window_{image_name}', width=width, height=height, pos=pos, no_scrollbar=False, no_resize=True, user_data={'image_name': image_name, 'hists_toggled': False, 'history_toggled': False, 'selections': []}, on_close=lambda: dpg.delete_item(window)) as window:
            with dpg.menu_bar():
                dpg.add_menu_item(label='Save', user_data=image_name, callback=lambda s, ad, ud: trigger_save_image_dialog(ud))
                build_transformations_menu(image_name)
                dpg.add_menu_item(label='Histograms', tag=f'hists_toggle_{image_name}', user_data=image_name, callback=lambda s, ad, ud: toggle_hists(ud))
                dpg.add_menu_item(label='History', tag=f'history_toggle_{image_name}', user_data=image_name, callback=lambda s, ad, ud: toggle_history(ud))
                dpg.add_menu_item(label='Ss', tag=f'push_selection_button_{image_name}', user_data=image_name, callback=lambda s, ad, ud: push_selection(ud))
                dpg.add_menu_item(label='Ds', tag=f'pop_selection_button_{image_name}', user_data=image_name, callback=lambda s, ad, ud: pop_selection(ud))

            with dpg.group(horizontal=True):
                with dpg.group():
                    image_item = dpg.add_image(image_name, tag=f'image_{image_name}', width=image.width, height=image.height)
                    # image_item = dpg.add_image('synth.jpg', tag=f'image_{image_name}', width=image.width, height=image.height)
                    render_image_overlay(image, window, image_item)
                    if movie:
                        render_movie_controls(movie, image.width)
                    with dpg.group(horizontal=True, width=image.width):
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

                with dpg.group(tag=f'history_group_{image.name}', width=HISTORY_WIDTH, show=False):
                    for i, tr in enumerate(image.transformations):
                        dpg.add_text(f'{i}. {tr}', tag=f'image_{image_name}_transformations_{i}')

def calculate_image_window_size(image: Image) -> Tuple[int, int]:
    # 120 = menu_bar + info_size, 10 = padding
    height = max(image.height, MIN_IMAGE_HEIGHT) + 120 + 10
    if image.movie_frame:
        height += 20  # Control buttons

    # 15 = padding
    width = max(image.width, MIN_IMAGE_WIDTH) + 15

    return width, height

def render_image_overlay(image: Image, window: Union[int, str], image_item: Union[int, str]):
    # IMPORTANTE: Para conseguir la posicion del image_item tenemos que darle a dpg un frame para que lo renderee
    dpg.split_frame()
    pos = dpg.get_item_pos(image_item)

    for tr in image.transformations:
        for tr_channel in tr.channel_transformations:
            if tr_channel.overlay:
                for cmd in tr_channel.overlay:
                    if isinstance(cmd, LineDrawCmd):
                        dpg.draw_line((cmd.p1_x, cmd.p1_y), (cmd.p2_x, cmd.p2_y), color=cmd.color, parent=window)

                    elif isinstance(cmd, CircleDrawCmd):
                        dpg.draw_circle((cmd.c_x, cmd.c_y), cmd.r, color=cmd.color, parent=window)

                    elif isinstance(cmd, ScatterDrawCmd):
                        if cmd.points.size == 0:
                            continue
                        mask = np.zeros((image.height, image.width, 4))
                        mask[cmd.points[:, 0], cmd.points[:, 1]] = np.array([*cmd.color, 255]) / 255
                        mask_tag = dpg.add_static_texture(image.width, image.height, mask.flatten(), parent=TEXTURE_REGISTRY)  # noqa

                        dpg.add_image(mask_tag, width=image.width, height=image.height, pos=pos, parent=window)

                    else:
                        raise NotImplementedError()

def render_movie_controls(movie: Movie, image_width: int):
    name = movie.name
    fr = movie.current_frame
    b1, b2, b3, b4 = f'movie_{name}_{fr}_ctrl_b1', f'movie_{name}_{fr}_ctrl_b2', f'movie_{name}_{fr}_ctrl_b3', f'movie_{name}_{fr}_ctrl_b4'
    width = image_width // 4

    @render_error
    def movie_control_callback(movie_name: str, frame: int):
        if not movie.on_first_frame():
            dpg.disable_item(b1)
            dpg.disable_item(b2)
        if not movie.on_last_frame():
            dpg.disable_item(b3)
            dpg.disable_item(b4)
        render_movie_frame(movie_name, frame)

    with dpg.group(horizontal=True):
        if not movie.on_first_frame():
            dpg.add_button(
                tag=b1,
                label='<<- 10',
                width=width - 1,
                user_data=(movie.name, movie.current_frame - 10),
                callback=lambda s, ad, ud: movie_control_callback(*ud),
            )
            dpg.add_button(
                tag=b2,
                label='<- Step',
                indent=width + 1,
                width=width - 1,
                user_data=(movie.name, movie.current_frame - 1),
                callback=lambda s, ad, ud: movie_control_callback(*ud)
            )
        if not movie.on_last_frame():
            dpg.add_button(
                tag=b3,
                label='Step ->',
                indent=(width + 1) * 2,
                width=width - 1,
                user_data=(movie.name, movie.current_frame + 1),
                callback=lambda s, ad, ud: movie_control_callback(*ud)
            )
            dpg.add_button(
                tag=b4,
                label='10 ->>',
                indent=(width + 1) * 3,
                width=width - 1,
                user_data=(movie.name, movie.current_frame + 10),
                callback=lambda s, ad, ud: movie_control_callback(*ud)
            )

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
        dpg.set_item_label(toggle, 'Histograms')
        dpg.hide_item(hists)
    else:
        dpg.set_item_label(toggle, 'Hide Histograms')
        dpg.show_item(hists)
    dpg.set_item_width(window, dpg.get_item_width(window) + diff)
    user_data['hists_toggled'] = not plots_toggled
    if not plots_toggled and user_data['history_toggled']:
        toggle_history(image_name)

@render_error
def toggle_history(image_name: str) -> None:
    window = f'image_window_{image_name}'
    toggle = f'history_toggle_{image_name}'
    hists = f'history_group_{image_name}'
    user_data = dpg.get_item_user_data(window)
    history_toggled: bool = user_data['history_toggled']
    diff: int = HISTORY_WIDTH + 10
    if history_toggled:
        diff = -diff
        dpg.set_item_label(toggle, 'History')
        dpg.hide_item(hists)
    else:
        dpg.set_item_label(toggle, 'Hide History')
        dpg.show_item(hists)
    dpg.set_item_width(window, dpg.get_item_width(window) + diff)
    user_data['history_toggled'] = not history_toggled
    if not history_toggled and user_data['hists_toggled']:
        toggle_hists(image_name)

@render_error
def push_selection(image_name: str) -> None:
    window = f'image_window_{image_name}'
    user_data = dpg.get_item_user_data(window)
    selections = user_data['selections']
    if 'init_draw' not in user_data or 'end_draw' not in user_data:
        return None

    p1, p2 = user_data['init_draw'], user_data['end_draw']

    p1_ret = int(min(p2[1], p1[1])), int(min(p2[0], p1[0]))
    p2_ret = int(max(p2[1], p1[1])), int(max(p2[0], p1[0]))

    if p1_ret[0] == p2_ret[0] or p1_ret[1] == p2_ret[1]:
        # Son colineares => No hay rectangulo
        return None

    selections.append((p1_ret, p2_ret))
    current_selection = f'image_{image_name}_selection'
    saved_selection = f'saved_selection_{p1_ret}_{p2_ret}'

    if dpg.does_item_exist(current_selection):
        dpg.delete_item(current_selection)

    dpg.draw_rectangle(p1, p2, parent=window, tag=saved_selection, color=(0x00, 0xCC, 0x00, 200))
    
    user_data.pop('init_draw', None)
    user_data.pop('end_draw', None)

@render_error
def pop_selection(image_name: str) -> None:
    window = f'image_window_{image_name}'
    user_data = dpg.get_item_user_data(window)
    selections = user_data['selections']

    p1_ret, p2_ret = selections.pop()
    
    saved_selection = f'saved_selection_{p1_ret}_{p2_ret}'

    if dpg.does_item_exist(saved_selection):
        dpg.delete_item(saved_selection)
    


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
    with dpg.window(label=f'Histogram - {image_name} - {theme.split("_")[0].title()} Channel'):
        build_histogram_plot(theme, hist.tolist(), bins.tolist()) # noqa - Hackazo para evitar el bug de que se renderea mal

def register_image(image: Image) -> None:
    image_vector = image_to_rgba_array(image)
    dpg.add_static_texture(image.width, image.height, image_vector, tag=image.name, parent=TEXTURE_REGISTRY) # noqa
    if not image.movie_frame:
        dpg.add_menu_item(label=image.name, parent=IMAGES_MENU, user_data=image.name, callback=lambda s, ad, ud: render_image_window(ud))

def register_movie(movie_name: str):
    dpg.add_menu_item(label=movie_name, parent=IMAGES_MENU, user_data=(movie_name, 0), callback=lambda s, ad, ud: render_movie_frame(*ud))

@render_error
def render_movie_frame(movie_name: str, frame: int):
    movie = mov_repo.get_movie(movie_name)
    frame_window = f'image_window_{movie.current_frame_name}'
    frame_is_rendered = dpg.does_item_exist(frame_window)
    frame_window_pos = dpg.get_item_pos(frame_window) if frame_is_rendered else []

    # Actualizamos la current frame de la movie
    frame_image: Image
    if movie.current_frame == frame:
        # Si ya estamos parados bien, solo tenemos que cargarla
        frame_image = load_current_movie_frame(movie)
    elif movie.current_frame > frame:
        # Si vamos para atras podemos saltar de una
        movie.current_frame = max(0, frame)
        frame_image = load_current_movie_frame(movie)
    else:
        # Cuando vamos para adelante estamos obligados a ir uno a uno para ir calculando el frame a partir del anterior
        frame_image = load_current_movie_frame(movie)
        for _ in range(movie.current_frame + 1, min(frame + 1, len(movie))):
            movie.inc_frame()
            frame_image = load_current_movie_frame(movie)

    render_image_window(frame_image.name, movie, pos=frame_window_pos)
    if frame_is_rendered:
        dpg.delete_item(frame_window)

def load_current_movie_frame(movie: Movie) -> Image:
    return load_movie_frame(movie, movie.current_frame)

# Recursively load image transformation chain
def load_movie_frame(movie: Movie, frame: int) -> Image:
    from models.movie import RootMovie, TransformedMovie

    frame_name = movie.get_frame_name(frame)
    if img_repo.contains_image(frame_name):
        return img_repo.get_image(frame_name)
    else:
        image: Image
        if isinstance(movie, RootMovie):
            image = load_image(movie.get_frame_path(frame), movie.name)
        elif isinstance(movie, TransformedMovie):
            prev_image = load_movie_frame(mov_repo.get_movie(movie.base_movie), frame)
            image = movie.last_transformation.inductive_handle(frame_name, frame, img_repo.get_image(movie.get_frame_name(frame - 1)), prev_image)
            image.movie = movie.name
        else:
            raise NotImplementedError()

        img_repo.persist_image(image)
        register_image(image)
        return image

@render_error
def load_image_handler(app_data):
    from models.movie import RootMovie

    movie_dir_checkbox = 'load_image_movie_dir_check'
    movie_dir: bool = dpg.get_value(movie_dir_checkbox)
    dpg.set_value(movie_dir_checkbox, False)

    current_path = app_data['current_path']
    selections: Dict[str, str] = movie_dir_selections(current_path) if movie_dir else app_data['selections']
    selection_count = len(selections)
    if selection_count == 0:
        raise ValueError('No image selected')

    image: Image
    movie: Optional[Movie]
    if selection_count == 1:
        # Caso Image
        image_path = next(iter(selections.values()))
        image_name = Image.name_from_path(image_path)

        if img_repo.contains_image(image_name):
            image = img_repo.get_image(image_name)
        else:
            image = load_image(image_path)
            img_repo.persist_image(image)
            register_image(image)

        movie = None
    else:
        # Caso Movie
        # WARNING: Identificamos a las peliculas por su directorio -> No pueden haber 2 peliculas en el mismo dir (sensato para nuestro caso de uso)
        movie_name = os.path.basename(current_path)

        if mov_repo.contains_movie(movie_name):
            movie = mov_repo.get_movie(movie_name)
        else:
            movie_base_path = os.path.dirname(app_data['file_path_name'])
            frames = list(selections.keys())
            frames.sort()  # WARNING: Las frames tienen que estar ordenadas alfabeticamente!!
            movie = RootMovie(movie_name, frames, movie_base_path)
            mov_repo.persist_movie(movie)
            register_movie(movie_name)

        image = load_current_movie_frame(movie)

    render_image_window(image.name, movie)

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
    dpg.add_file_dialog(tag=SAVE_IMAGE_DIALOG, default_path='../images', directory_selector=True, show=False, modal=True, width=1024, height=512, callback=lambda s, ad, ud: save_image_handler(ad, ud))

def build_load_image_dialog(default_path: str) -> None:
    with dpg.file_dialog(label='Choose file to load...', tag=LOAD_IMAGE_DIALOG, default_path=default_path, directory_selector=False, show=False, modal=True, width=1024, height=512, callback=lambda s, ad: load_image_handler(ad)):
        dpg.add_file_extension(f'Image{{{",".join(valid_image_formats())}}}')
        with dpg.group(horizontal=True):
            dpg.add_text('Select Movie Directory')
            dpg.add_checkbox(default_value=False, tag='load_image_movie_dir_check')

def build_load_metadata_dialog() -> None:
    with dpg.file_dialog(label='Choose metadata file to load...', tag=LOAD_METADATA_DIALOG, default_path='../images', directory_selector=False, show=False, modal=True, width=1024, height=512, callback=lambda s, ad: load_metadata_handler(ad)):
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

@render_error
def close_all_windows() -> None:
    for win_id in dpg.get_windows():
        win = dpg.get_item_alias(win_id)
        if is_image_window(win):
            dpg.delete_item(win)

# Retorna los puntos ya normalizados
# Formato: (y, x); (up_left, down_right); Acordarse: left < right, up < down
def get_image_window_rect_selections(window: Union[str, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    user_data = dpg.get_item_user_data(window)
    return user_data['selections']

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
                if dpg.does_item_exist(pointer):
                    dpg.set_value(pointer, '')

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
                if 'init_draw' in usr_data and 'end_draw' not in usr_data:
                    if dpg.does_item_exist(selection):
                        dpg.delete_item(selection)

                    dpg.draw_rectangle(usr_data['init_draw'], pixel, parent=window, tag=selection, color=(0xCC, 0x00, 0x66, 200))

                dpg.show_item(pointer)
                dpg.set_value(pointer, f"Pixel: {get_pixel_pos_in_image(window, image_name)}  Value: {np.around(image.get_pixel(pixel), 2)}")
                return  # Terminamos de dibujar

        # No estamos en la imagen -> Borramos el puntero. El rectangulo lo dejamos
        dpg.set_value(pointer, '')

    @render_error
    def mouse_down_handler():
        window = get_focused_hovered_image_window()

        # Cleanup de las otras image windows
        # for win_id in dpg.get_windows():
        #     win = dpg.get_item_alias(win_id)
        #     if window != win and is_image_window(win):
        #       No cleanupeamos nada. Queremos que el rectangulo sea persistente

        if window == 0:
            return  # No hay ninguna imagen seleccionada

        usr_data = dpg.get_item_user_data(window)
        image_name: str = usr_data['image_name']
        pixel_pos = get_pixel_pos_in_image(window, image_name)

        if pixel_pos[0] < 0 or pixel_pos[1] < 0:
          return

        usr_data['init_draw'] = pixel_pos if ('init_draw' not in usr_data or 'end_draw' in usr_data) else usr_data['init_draw']

        usr_data.pop('end_draw', None)
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

        if 'init_draw' not in usr_data or 'end_draw' in usr_data:
            return  # No se detecto el inicio del click (probablemente se este arrastrando la ventana), o ya procesamos el rectangulo

        usr_data['end_draw'] = get_pixel_pos_in_image(window, image_name)
        init_draw = usr_data['init_draw']
        end_draw = usr_data['end_draw']

        x = (int(min(end_draw[0], init_draw[0])), int(max(end_draw[0], init_draw[0])))
        y = (int(min(end_draw[1], init_draw[1])), int(max(end_draw[1], init_draw[1])))

        if x[1] > x[0] and y[1] > y[0]:
            if image.channels > 1:
                # TODO: A veces lanza una excepcion: 'RuntimeWarning: Mean of empty slice.'
                mean = np.mean(np.array(image.data[y[0]:y[1], x[0]:x[1]]).reshape((-1, 3)), axis=0)
            else:
                mean = np.mean(image.data[y[0]:y[1], x[0]:x[1]])

            dpg.set_value(region, f"#Pixel: {np.prod((y[1] - y[0] + 1, x[1] - x[0] + 1))}  Avg: {np.around(mean, 2)}")
            dpg.show_item(region)
        else:
            dpg.set_value(region, '')
            if dpg.does_item_exist(selection):
                dpg.delete_item(selection)

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=mouse_move_handler)
        dpg.add_mouse_down_handler(callback=mouse_down_handler)
        dpg.add_mouse_release_handler(callback=mouse_release_handler)
