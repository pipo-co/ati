import itertools
from typing import Callable, List, Optional, Any, Tuple

import dearpygui.dearpygui as dpg
import numpy as np

from models.movie import TransformedMovie, MovieTransformation
from transformations import basic, border, combine, denoise, noise, threshold as thresh
from repositories import images_repo as img_repo
from repositories import movies_repo as mov_repo
from transformations.data_models import LinRange
from . import interface
from models.image import Image, ImageTransformation, strip_extension, get_extension, ImageFormat, normalize, MAX_COLOR, \
    ImageChannelTransformation
from frontend.interface_utils import render_error
from transformations.sliding import PaddingStrategy

# General Items
TR_DIALOG: str = 'tr_dialog'

# Custom Inputs
TR_NAME_INPUT: str              = 'tr_name_input'
TR_IMG_INPUT: str               = 'tr_img_input'
TR_INT_VALUE_SELECTOR: str      = 'tr_int_value_input'
TR_FLOAT_VALUE_SELECTOR: str    = 'tr_float_value_input'
TR_INT_LIST_VALUE_SELECTOR: str = 'tr_int_list_value_input'
TR_RANGE_VALUE_SELECTOR: str    = 'tr_range_value_input'
TR_RADIO_BUTTONS: str           = 'tr_radio_buttons'
TR_CHECKBOX: str                = 'tr_checkbox'
TR_INT_TABLE: str               = 'tr_int_table'

TR_DECIMAL_PLACES = 2
CHANNELS = ['Red', 'Green', 'Blue']

TrHandler = Callable[[str], Image]

def build_transformations_menu(image_name: str) -> None:
    with dpg.menu(label='Transform'):
        with dpg.menu(label='Utils'):
            build_tr_menu_item(TR_COPY,                     build_copy_dialog,                      image_name)
            build_tr_menu_item(TR_REFORMAT,                 build_reformat_dialog,                  image_name)
            build_tr_menu_item(TR_NORMALIZE,                build_normalize_dialog,                 image_name)
            build_tr_menu_item(TR_SLICE,                    build_slice_dialog,                     image_name)
        with dpg.menu(label='Basic'):
            build_tr_menu_item(TR_NEG,                      build_neg_dialog,                       image_name)
            build_tr_menu_item(TR_POW,                      build_pow_dialog,                       image_name)
            build_tr_menu_item(TR_EQUALIZE,                 build_equalize_dialog,                  image_name)
        with dpg.menu(label='Threshold'):
            build_tr_menu_item(TR_THRESH_MANUAL,            build_thresh_manual_dialog,             image_name)
            build_tr_menu_item(TR_THRESH_GLOBAL,            build_thresh_global_dialog,             image_name)
            build_tr_menu_item(TR_THRESH_OTSU,              build_thresh_otsu_dialog,               image_name)
        with dpg.menu(label='Noise'):
            build_tr_menu_item(TR_NOISE_GAUSS,              build_noise_gauss_dialog,               image_name)
            build_tr_menu_item(TR_NOISE_EXP,                build_noise_exp_dialog,                 image_name)
            build_tr_menu_item(TR_NOISE_RAYLEIGH,           build_noise_rayleigh_dialog,            image_name)
            build_tr_menu_item(TR_NOISE_SALT,               build_noise_salt_dialog,                image_name)
        with dpg.menu(label='Denoise'):
            build_tr_menu_item(TR_DENOISE_MEAN,             build_denoise_mean_dialog,              image_name)
            build_tr_menu_item(TR_DENOISE_MEDIAN,           build_denoise_median_dialog,            image_name)
            build_tr_menu_item(TR_DENOISE_WEIGHTED_MEDIAN,  build_denoise_weighted_median_dialog,   image_name)
            build_tr_menu_item(TR_DENOISE_GAUSS,            build_denoise_gauss_dialog,             image_name)
            build_tr_menu_item(TR_DENOISE_DIFFUSION,        build_denoise_diffusion_dialog,         image_name)
            build_tr_menu_item(TR_DENOISE_BILATERAL,        build_denoise_bilateral_dialog,         image_name)
        with dpg.menu(label='Border'):
            build_tr_menu_item(TR_BORDER_DIRECTIONAL,       build_border_directional_dialog,        image_name)
            build_tr_menu_item(TR_BORDER_HIGH_PASS,         build_border_high_pass_dialog,          image_name)
            build_tr_menu_item(TR_BORDER_PREWITT,           build_border_prewitt_dialog,            image_name)
            build_tr_menu_item(TR_BORDER_SOBEL,             build_border_sobel_dialog,              image_name)
            build_tr_menu_item(TR_BORDER_LAPLACIAN,         build_border_laplacian_dialog,          image_name)
            build_tr_menu_item(TR_BORDER_LOG,               build_border_log_dialog,                image_name)
            build_tr_menu_item(TR_BORDER_SUSAN,             build_border_susan_dialog,              image_name)
            build_tr_menu_item(TR_BORDER_HOUGH_LINE,        build_border_hough_line_dialog,         image_name)
            build_tr_menu_item(TR_BORDER_HOUGH_CIRCLE,      build_border_hough_circle_dialog,       image_name)
            build_tr_menu_item(TR_BORDER_CANNY,             build_border_canny_dialog,              image_name)
            build_tr_menu_item(TR_BORDER_HARRIS,            build_border_harris_dialog,             image_name)
            build_tr_menu_item(TR_BORDER_ACTIVE_OUTLINE,    build_border_active_outline_dialog,     image_name)
            build_tr_menu_item(TR_BORDER_MULTIPLE_ACTIVE_OUTLINE,    build_border_multiple_active_outline_dialog,     image_name)
        with dpg.menu(label='Combine'):
            build_tr_menu_item(TR_COMBINE_ADD,              build_combine_add_dialog,               image_name)
            build_tr_menu_item(TR_COMBINE_SUB,              build_combine_sub_dialog,               image_name)
            build_tr_menu_item(TR_COMBINE_MULT,             build_combine_mult_dialog,              image_name)
            build_tr_menu_item(TR_COMBINE_SIFT,             build_combine_sift_dialog,              image_name)


def build_tr_menu_item(tr_id: str, tr_dialog_builder: Callable[[str], None], image_name: str) -> None:
    dpg.add_menu_item(label=tr_id.title().replace('_', ' '), user_data=(tr_dialog_builder, image_name), callback=lambda s, ad, ud: ud[0](ud[1]))

@render_error
def execute_image_transformation(image_name: str, handler: TrHandler) -> None:
    try:
        new_image = handler(image_name)
    finally:
        dpg.delete_item(TR_DIALOG)

    img_repo.persist_image(new_image)
    interface.register_image(new_image)
    interface.render_image_window(new_image.name)

@render_error
def execute_movie_transformation(base_movie_name: str, base_handle: TrHandler, inductive_handle: Callable[[str, int, Image, Image], Image]) -> None:
    base_movie = mov_repo.get_movie(base_movie_name)
    base_image = interface.load_movie_frame(base_movie, 0)

    try:
        new_image = base_handle(base_image.name)
    finally:
        dpg.delete_item(TR_DIALOG)

    movie_name = strip_extension(new_image.name)
    image_tr = new_image.last_transformation

    new_movie = TransformedMovie(movie_name, base_movie, MovieTransformation.from_img_tr(image_tr, inductive_handle))

    mov_repo.persist_movie(new_movie)
    interface.register_movie(new_movie.name)

    new_image.name = new_movie.get_frame_name(0)
    new_image.movie = movie_name

    img_repo.persist_image(new_image)
    interface.register_image(new_image)

    interface.render_image_window(new_image.name, new_movie)

def build_tr_dialog(tr_id: str) -> int:
    return dpg.window(label=f'Apply {tr_id.title().replace("_", " ")} Transformation', tag=TR_DIALOG, modal=True, no_close=True, pos=interface.CENTER_POS)

# Generic inductive handler that takes all inputs from previous movie frame
def generic_tr_inductive_handle(fn: Callable[[Image, Any], Tuple[np.ndarray, List[ImageChannelTransformation]]]) -> Callable[[str, int, Image, Image], Image]:
    def ret(new_name: str, _: int, prev: Image, image: Image) -> Image:
        new_data, channels_tr = fn(image, **prev.all_inputs())
        return image.transform(new_name, new_data, ImageTransformation(prev.last_transformation.name, prev.major_inputs, prev.minor_inputs, channels_tr))
    return ret

# Also allows to transform a movie
def build_tr_dialog_end_buttons(tr_id: str, image_name: str, handle: TrHandler, inductive_handle: Optional[Callable[[str, int, Image, Image], Image]]) -> None:
    image = img_repo.get_image(image_name)

    with dpg.group(horizontal=True):
        dpg.add_button(label='Transform Image', user_data=(image_name, handle), callback=lambda s, ap, ud: execute_image_transformation(*ud))
        if image.movie and inductive_handle is not None:
            dpg.add_button(label='Transform Movie', user_data=(image.movie, handle, inductive_handle), callback=lambda s, ap, ud: execute_movie_transformation(*ud))
        dpg.add_button(label='Cancel', user_data=tr_id, callback=lambda: dpg.delete_item(TR_DIALOG))

# ******************** Input Builders ********************* #

def unique_name(base_name: str, ext: str = '') -> str:
    if not img_repo.contains_image(base_name + ext) and not mov_repo.contains_movie(base_name):
        return base_name
    for i in itertools.count(start=2):
        name = f'{base_name}({i})'
        if not img_repo.contains_image(name + ext) and not mov_repo.contains_movie(name):
            return name

# Solo puede haber un name input, que (casi) siempre debe estar
def build_tr_name_input(tr_id: str, image_name: str) -> None:
    image = img_repo.get_image(image_name)
    default_value = unique_name(image.movie + f'_{tr_id}') \
        if image.movie_frame \
        else unique_name(strip_extension(image_name) + f'_{tr_id}', get_extension(image_name))

    dpg.add_text('Select New Image or Movie Name (no extension)')
    dpg.add_input_text(default_value=default_value, tag=TR_NAME_INPUT)

def build_tr_value_int_selector(name: str, min_val: int, max_val: int, default_value: int = None, suffix: str = None, step: int = 1, tag: str = TR_INT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select \'{name}\' between {min_val} and {max_val}')
    dpg.add_input_int(min_value=min_val, max_value=max_val, default_value=min_val if default_value is None else default_value, step=step, label=suffix, tag=tag)

def build_tr_value_float_selector(name: str, min_val: float, max_val: float, default_value: float = None, suffix: str = None, tag: str = TR_FLOAT_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select \'{name}\' between {min_val} and {max_val}')
    dpg.add_input_float(min_value=min_val, max_value=max_val, default_value=min_val if default_value is None else default_value, label=suffix, tag=tag)
    
def build_tr_value_int_list_selector(name: str, min_val: float, max_val: float, default_value: int = None, tag: str = TR_INT_LIST_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select one or more  \'{name}\' between {min_val} and {max_val} separated by ,')
    dpg.add_input_text(default_value=str(default_value), tag=tag)

def build_tr_value_range_selector(name: str, min_val: float, max_val: float, max_count: int, tag: str = TR_RANGE_VALUE_SELECTOR) -> None:
    dpg.add_text(f'Select a range for \'{name}\' between {min_val} and {max_val}. Max count: {max_count}')
    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()

        with dpg.table_row():
            dpg.add_input_float(min_value=min_val, max_value=max_val,   default_value=min_val,    step=0, label='min',  tag=f'{tag}_min_val')
            dpg.add_input_float(min_value=min_val, max_value=max_val,   default_value=max_val,    step=0, label='max',  tag=f'{tag}_max_val')
            dpg.add_input_int(  min_value=0,       max_value=max_count, default_value=max_count,  step=0, label='n',    tag=f'{tag}_count')

def build_tr_percentage_selector(name: str, default_value: int = 20, tag: str = TR_INT_VALUE_SELECTOR) -> None:
    build_tr_value_int_selector(name, 0, 100, default_value=default_value, suffix='%', tag=tag)

def build_tr_radio_buttons(names: List[str], default_value: Optional[str] = None, horizontal: bool = True, tag: str = TR_RADIO_BUTTONS) -> None:
    names = list(map(lambda s: s.replace('_', ' ').title(), names))
    if default_value:
        default_value = default_value.title()
    else:
        default_value = names[0]
    dpg.add_radio_button(items=names, default_value=default_value, horizontal=horizontal, tag=tag)

def build_tr_checkbox(name: str, default_value: bool = False, tag: str = TR_CHECKBOX) -> None:
    dpg.add_checkbox(label=name, default_value=default_value, tag=tag)
    
def build_tr_img_selector(image_name: str, same_shape=False) -> None:
    image_list = list(map(lambda img: img.name, img_repo.get_same_shape_images(image_name) if same_shape else img_repo.get_images(image_name)))
    dpg.add_text('Select Another Image to combine')
    dpg.add_listbox(image_list, tag=TR_IMG_INPUT)

def build_tr_input_table(size: int = 3, tag: str = TR_INT_TABLE):
    with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()

        for row in range(size):
            with dpg.table_row():
                for col in range(size):
                    with dpg.table_cell():
                        cell_tag = get_table_field_name(tag, row, col)
                        dpg.add_input_int(min_value=1, max_value=50, step=0, tag=cell_tag)

def get_table_field_name(tag: str, row: int, col: int) -> str:
    return f'{tag}_{row}_{col}'

# ******************** Input Getters and Requirements ********************* #

def get_tr_name_value(image: Image) -> str:
    base_name = dpg.get_value(TR_NAME_INPUT)
    if not base_name:
        raise ValueError('A name for the new image/movie must be provided')
    ret = base_name + image.format.to_extension()
    if img_repo.contains_image(ret):
        raise ValueError(f'Another image with name "{ret}" already exists')
    if mov_repo.contains_movie(base_name):
        raise ValueError(f'Another movie with name "{base_name}" already exists')
    return ret

def get_tr_img_value(img_input: str = TR_IMG_INPUT) -> Image:
    img_name = dpg.get_value(img_input)
    if not img_repo.contains_image(img_name):
        raise ValueError('Selecting a valid image is required for transformation')
    return img_repo.get_image(img_name)

def get_tr_radio_buttons_value(radio_buttons: str = TR_RADIO_BUTTONS) -> str:
    return dpg.get_value(radio_buttons).replace(' ', '_')

def get_tr_checkbox_value(checkbox: str = TR_CHECKBOX) -> bool:
    return dpg.get_value(checkbox)

def get_tr_int_value(int_input: str = TR_INT_VALUE_SELECTOR) -> int:
    return dpg.get_value(int_input)

def get_tr_float_value(float_input: str = TR_FLOAT_VALUE_SELECTOR) -> float:
    return dpg.get_value(float_input)

def get_tr_int_list_value(list_input: str = TR_INT_LIST_VALUE_SELECTOR) -> List[int]:
    return list(map(lambda n: int(n), dpg.get_value(list_input).split(',')))

def get_tr_range_value(range_input: str = TR_RANGE_VALUE_SELECTOR) -> LinRange:
    min_v = dpg.get_value(f'{range_input}_min_val')
    max_v = dpg.get_value(f'{range_input}_max_val')
    count = dpg.get_value(f'{range_input}_count')
    return LinRange(min_v, max_v, count)

def get_tr_percentage_value(percentage_input: str = TR_FLOAT_VALUE_SELECTOR) -> int:
    percentage = get_tr_int_value(percentage_input)
    if not 0 <= percentage <= 100:
        raise ValueError('Percentage must be between 0 and 100')
    return percentage

def get_tr_input_table_values(size: int = 3, table_tag: str = TR_INT_TABLE) -> List[List[int]]:
    table_values = []
    for row in range(0, size):
        table_values.append([])
        for col in range(0, size):
            cell_tag = get_table_field_name(table_tag, row, col)
            table_values[row].append(dpg.get_value(cell_tag))

    return table_values

def require_odd(n: int, msg: str) -> int:
    if n % 2 == 0:
        raise ValueError(msg)
    return n

def require_same_shape(img1: Image, img2: Image, msg: str) -> None:
    if img1.shape != img2.shape:
        raise ValueError(msg)

########################################################
# ******************** Utilities ********************* #
########################################################

TR_COPY: str = 'copy'
@render_error
def build_copy_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COPY):
        build_tr_name_input(TR_COPY, image_name)
        # Aca declaramos inputs necesarios para el handle. Este caso no tiene.
        build_tr_dialog_end_buttons(TR_COPY, image_name, tr_copy, generic_tr_inductive_handle(lambda img: (img.data, [])))

def tr_copy(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, image.format, image.data)

TR_REFORMAT: str = 'reformat'
@render_error
def build_reformat_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COPY):
        image = img_repo.get_image(image_name)
        fmts = ImageFormat.values()
        if image.channels > 1:
            # Raw solo funciona para imagenes de un solo canal
            fmts.remove(ImageFormat.RAW.value)
        build_tr_radio_buttons(fmts)
        build_tr_dialog_end_buttons(TR_REFORMAT, image_name, tr_reformat, None)

def tr_reformat(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_fmt  = ImageFormat.from_str(get_tr_radio_buttons_value().lower())
    new_name = strip_extension(image_name) + new_fmt.to_extension()
    # 2. Procesamos
    # Do Nothing
    # 3. Creamos Imagen
    return Image(new_name, new_fmt, image.data)

TR_NORMALIZE: str = 'normalize'
@render_error
def build_normalize_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NORMALIZE):
        build_tr_name_input(TR_NORMALIZE, image_name)
        build_tr_dialog_end_buttons(TR_NORMALIZE, image_name, tr_normalize, generic_tr_inductive_handle(lambda img: (normalize(img.data, np.float64), [])))

def tr_normalize(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    new_data = normalize(image.data, np.float64)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_NORMALIZE, {}, {}))

TR_SLICE: str = 'slice'
@render_error
def build_slice_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_SLICE):
        build_tr_name_input(TR_SLICE, image_name)
        build_tr_radio_buttons(CHANNELS)
        build_tr_dialog_end_buttons(TR_SLICE, image_name, tr_slice, generic_tr_inductive_handle(basic.slice_channel))

def tr_slice(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    if image.channels == 1:
        return image.transform(new_name, image.data, ImageTransformation(TR_SLICE, {}, {}, []))
    channel   = CHANNELS.index(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = basic.slice_channel(image, channel)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_SLICE, {}, {}, channels_tr))

########################################################
# ********************** Basic *********************** #
########################################################

TR_NEG: str = 'neg'
@render_error
def build_neg_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NEG):
        build_tr_name_input(TR_NEG, image_name)
        build_tr_dialog_end_buttons(TR_NEG, image_name, tr_neg, generic_tr_inductive_handle(basic.negate))

def tr_neg(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    new_data, channels_tr = basic.negate(image)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_NEG, {}, {}, channels_tr))

TR_POW: str = 'pow'
@render_error
def build_pow_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_POW):
        build_tr_name_input(TR_POW, image_name)
        build_tr_value_float_selector('gamma', 0, 2)
        build_tr_dialog_end_buttons(TR_POW, image_name, tr_pow, generic_tr_inductive_handle(basic.power))

def tr_pow(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    gamma    = get_tr_float_value()
    # 2. Procesamos
    new_data, channels_tr = basic.power(image, gamma)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_POW, {'gamma': gamma}, {}, channels_tr))

TR_EQUALIZE: str = 'equalize'
@render_error
def build_equalize_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_EQUALIZE):
        build_tr_name_input(TR_EQUALIZE, image_name)
        build_tr_dialog_end_buttons(TR_EQUALIZE, image_name, tr_equalize, generic_tr_inductive_handle(basic.equalize))

def tr_equalize(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image    = img_repo.get_image(image_name)
    new_name = get_tr_name_value(image)
    # 2. Procesamos
    new_data, channels_tr = basic.equalize(image)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_EQUALIZE, {}, {}, channels_tr))

########################################################
# ******************** Threshold ********************* #
########################################################

TR_THRESH_MANUAL: str = 'manual'
@render_error
def build_thresh_manual_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_THRESH_MANUAL):
        build_tr_name_input(TR_THRESH_MANUAL, image_name)
        build_tr_value_int_selector('threshold', 0, MAX_COLOR, default_value=15)
        build_tr_dialog_end_buttons(TR_THRESH_MANUAL, image_name, tr_thresh_manual, generic_tr_inductive_handle(thresh.manual))

def tr_thresh_manual(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    threshold   = get_tr_int_value()
    # 2. Procesamos
    new_data, channels_tr = thresh.manual(image, threshold)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_THRESH_MANUAL, {'threshold': threshold}, {}, channels_tr))

TR_THRESH_GLOBAL: str = 'global'
@render_error
def build_thresh_global_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_THRESH_GLOBAL):
        build_tr_name_input(TR_THRESH_GLOBAL, image_name)
        build_tr_value_int_selector('starting threshold', 0, MAX_COLOR, default_value=MAX_COLOR//2)
        build_tr_dialog_end_buttons(TR_THRESH_GLOBAL, image_name, tr_global_umbral, generic_tr_inductive_handle(thresh.global_))

def tr_global_umbral(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    threshold   = get_tr_int_value()
    # 2. Procesamos
    new_data, channels_tr = thresh.global_(image, threshold)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_THRESH_MANUAL, {'threshold': threshold}, {}, channels_tr))

TR_THRESH_OTSU: str = 'otsu'
@render_error
def build_thresh_otsu_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_THRESH_OTSU):
        build_tr_name_input(TR_THRESH_OTSU, image_name)
        build_tr_dialog_end_buttons(TR_THRESH_OTSU, image_name, tr_otsu_threshold, generic_tr_inductive_handle(thresh.otsu))

def tr_otsu_threshold(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    # 2. Procesamos
    new_data, channels_tr = thresh.otsu(image)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_THRESH_OTSU, {}, {}, channels_tr))

########################################################
# ********************** Noise *********************** #
########################################################

TR_NOISE_GAUSS: str = 'gauss'
@render_error
def build_noise_gauss_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_GAUSS):
        build_tr_name_input(TR_NOISE_GAUSS, image_name)
        build_tr_radio_buttons(noise.Type.names(), default_value=noise.Type.ADDITIVE.name)
        build_tr_value_float_selector('sigma', 0, 1, default_value=0.1, tag='sigma')
        build_tr_percentage_selector('noise percentage', tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_GAUSS, image_name, tr_noise_gauss, generic_tr_inductive_handle(noise.gauss))

def tr_noise_gauss(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sigma       = get_tr_float_value('sigma')
    percentage  = get_tr_percentage_value('percentage')
    noise_type  = noise.Type.from_name(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = noise.gauss(image, sigma, noise_type, percentage)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_NOISE_GAUSS, {'sigma': sigma, 'noise_type': noise_type, 'percentage': percentage}, {}, channels_tr))

TR_NOISE_EXP: str = 'exp'
@render_error
def build_noise_exp_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_EXP):
        build_tr_name_input(TR_NOISE_EXP, image_name)
        build_tr_radio_buttons(noise.Type.names(), default_value=noise.Type.MULTIPLICATIVE.name)
        build_tr_value_float_selector('lambda', 1, 5, default_value=3, tag='lambda')
        build_tr_percentage_selector('noise percentage', tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_EXP, image_name, tr_noise_exp, generic_tr_inductive_handle(noise.exponential))

def tr_noise_exp(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    lam         = get_tr_float_value('lambda')
    percentage  = get_tr_percentage_value('percentage')
    noise_type  = noise.Type.from_name(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = noise.exponential(image, lam, noise_type, percentage)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_NOISE_EXP, {'lam': lam, 'noise_type': noise_type, 'percentage': percentage}, {}, channels_tr))

TR_NOISE_RAYLEIGH: str = 'rayleigh'
@render_error
def build_noise_rayleigh_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_RAYLEIGH):
        build_tr_name_input(TR_NOISE_RAYLEIGH, image_name)
        build_tr_radio_buttons(noise.Type.names(), default_value=noise.Type.MULTIPLICATIVE.name)
        build_tr_value_float_selector('epsilon', 0, 1, default_value=0.6, tag='epsilon')
        build_tr_percentage_selector('noise percentage', tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_RAYLEIGH, image_name, tr_noise_rayleigh, generic_tr_inductive_handle(noise.rayleigh))

def tr_noise_rayleigh(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    epsilon     = dpg.get_value('epsilon')
    percentage  = dpg.get_value('percentage')
    noise_type  = noise.Type.from_name(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = noise.rayleigh(image, epsilon, noise_type, percentage)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_NOISE_RAYLEIGH, {'epsilon': epsilon, 'noise_type': noise_type, 'percentage': percentage}, {}, channels_tr))

TR_NOISE_SALT: str = 'salt'
@render_error
def build_noise_salt_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_NOISE_SALT):
        build_tr_name_input(TR_NOISE_SALT, image_name)
        build_tr_percentage_selector('salt percentage', default_value=5, tag='percentage')
        build_tr_dialog_end_buttons(TR_NOISE_SALT, image_name, tr_noise_salt, generic_tr_inductive_handle(noise.salt))

def tr_noise_salt(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    percentage  = get_tr_percentage_value('percentage')
    # 2. Procesamos
    new_data, channels_tr = noise.salt(image, percentage)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_NOISE_SALT, {'percentage': percentage}, {}, channels_tr))

########################################################
# ********************* Denoise ********************** #
########################################################

TR_DENOISE_MEAN: str = 'mean'
@render_error
def build_denoise_mean_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_MEAN):
        build_tr_name_input(TR_DENOISE_MEAN, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_MEAN, image_name, tr_denoise_mean, generic_tr_inductive_handle(denoise.mean))

def tr_denoise_mean(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = denoise.mean(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_DENOISE_MEAN, {'kernel_size': kernel_size}, {'padding_str': padding_str}, channels_tr))

TR_DENOISE_MEDIAN: str = 'median'
@render_error
def build_denoise_median_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_MEDIAN):
        build_tr_name_input(TR_DENOISE_MEDIAN, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(denoise.PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_MEDIAN, image_name, tr_denoise_median, generic_tr_inductive_handle(denoise.median))

def tr_denoise_median(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = denoise.median(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_DENOISE_MEDIAN, {'kernel_size': kernel_size}, {'padding_str': padding_str}, channels_tr))

TR_DENOISE_WEIGHTED_MEDIAN: str = 'weighted median'
@render_error
def build_denoise_weighted_median_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_WEIGHTED_MEDIAN):
        build_tr_name_input(TR_DENOISE_WEIGHTED_MEDIAN, image_name)
        build_tr_radio_buttons(denoise.PaddingStrategy.names())
        build_tr_input_table()
        build_tr_dialog_end_buttons(TR_DENOISE_WEIGHTED_MEDIAN, image_name, tr_denoise_weighted_median, generic_tr_inductive_handle(denoise.weighted_median))

def tr_denoise_weighted_median(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel      = np.array(get_tr_input_table_values())
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos - Puede ser async
    new_data, channels_tr = denoise.weighted_median(image, kernel, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_DENOISE_WEIGHTED_MEDIAN, {}, {'kernel': kernel, 'padding_str': padding_str}, channels_tr))

TR_DENOISE_GAUSS: str = 'gauss'
@render_error
def build_denoise_gauss_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_GAUSS):
        build_tr_name_input(TR_DENOISE_GAUSS, image_name)
        build_tr_value_float_selector('sigma', 1, 7, default_value=3)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_GAUSS, image_name, tr_denoise_gauss, generic_tr_inductive_handle(denoise.gauss))

def tr_denoise_gauss(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sigma       = get_tr_float_value()
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = denoise.gauss(image, sigma, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_DENOISE_GAUSS, {'sigma': sigma}, {'padding_str': padding_str}, channels_tr))

TR_DENOISE_DIFFUSION: str = 'diffusion'
@render_error
def build_denoise_diffusion_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_DIFFUSION):
        build_tr_name_input(TR_DENOISE_DIFFUSION, image_name)
        build_tr_value_int_selector('iterations', 0, denoise.MAX_ANISOTROPIC_ITERATIONS, default_value=10)
        build_tr_value_int_selector('sigma', 1, 10, default_value=4, tag='sigma')
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_radio_buttons(denoise.DiffusionStrategy.names(), tag='function')
        build_tr_dialog_end_buttons(TR_DENOISE_DIFFUSION, image_name, tr_denoise_diffusion, generic_tr_inductive_handle(denoise.diffusion))

def tr_denoise_diffusion(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    iterations  = get_tr_int_value()
    sigma       = get_tr_int_value('sigma')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    function = denoise.DiffusionStrategy.from_str(get_tr_radio_buttons_value(radio_buttons='function'))
    # 2. Procesamos
    new_data, channels_tr = denoise.diffusion(image, iterations, sigma, padding_str, function)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_DENOISE_DIFFUSION, {'iterations': iterations, 'sigma': sigma, 'function': function}, {'padding_str': padding_str}, channels_tr))

TR_DENOISE_BILATERAL: str = 'bilateral'
@render_error
def build_denoise_bilateral_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_DENOISE_BILATERAL):
        build_tr_name_input(TR_DENOISE_BILATERAL, image_name)
        build_tr_value_int_selector('sigma space', 0, 10, default_value=8, tag='sigma_space')
        build_tr_value_float_selector('sigma intensity', 0, 20, default_value=4, tag='sigma_intensity')
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_DENOISE_BILATERAL, image_name, tr_denoise_bilateral_filter, generic_tr_inductive_handle(denoise.bilateral))

def tr_denoise_bilateral_filter(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image           = img_repo.get_image(image_name)
    new_name        = get_tr_name_value(image)
    sigma_space     = get_tr_int_value('sigma_space')
    sigma_intensity = get_tr_float_value('sigma_intensity')
    padding_str     = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = denoise.bilateral(image, sigma_space, sigma_intensity, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_DENOISE_BILATERAL, {'sigma_space': sigma_space, 'sigma_intensity': sigma_intensity}, {'padding_str': padding_str}, channels_tr))

########################################################
# ********************** Border *********************** #
########################################################

TR_BORDER_HIGH_PASS: str = 'high pass'
@render_error
def build_border_high_pass_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_HIGH_PASS):
        build_tr_name_input(TR_BORDER_HIGH_PASS, image_name)
        build_tr_value_int_selector('kernel size', 3, 23, step=2)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_BORDER_HIGH_PASS, image_name, tr_border_high_pass, generic_tr_inductive_handle(border.high_pass))

def tr_border_high_pass(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    kernel_size = require_odd(get_tr_int_value(), 'Kernel size must be odd')
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = border.high_pass(image, kernel_size, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_HIGH_PASS, {'kernel_size': kernel_size}, {'padding_str': padding_str}, channels_tr))

TR_BORDER_DIRECTIONAL: str = 'directional'
@render_error
def build_border_directional_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_DIRECTIONAL):
        build_tr_name_input(TR_BORDER_DIRECTIONAL, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_radio_buttons(border.Direction.names(), tag="direction")
        build_tr_checkbox('Juliana\'s Kernel')
        build_tr_dialog_end_buttons(TR_BORDER_DIRECTIONAL, image_name, tr_border_directional, generic_tr_inductive_handle(border.directional))

def tr_border_directional(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    border_dir  = border.Direction.from_str(get_tr_radio_buttons_value(radio_buttons="direction"))
    kernel      = border.FamousKernel.JULIANA if get_tr_checkbox_value() else border.FamousKernel.PREWITT
    # 2. Procesamos
    new_data, channels_tr = border.directional(image, kernel, border_dir, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_DIRECTIONAL, {'border_dir': border_dir}, {'kernel': kernel, 'padding_str': padding_str}, channels_tr))

TR_BORDER_PREWITT: str = 'prewitt'
@render_error
def build_border_prewitt_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_PREWITT):
        build_tr_name_input(TR_BORDER_PREWITT, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_BORDER_PREWITT, image_name, tr_border_prewitt, generic_tr_inductive_handle(border.prewitt))

def tr_border_prewitt(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = border.prewitt(image, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_PREWITT, {}, {'padding_str': padding_str}, channels_tr))

TR_BORDER_SOBEL: str = 'sobel'
@render_error
def build_border_sobel_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_SOBEL):
        build_tr_name_input(TR_BORDER_SOBEL, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_BORDER_SOBEL, image_name, tr_border_sobel, generic_tr_inductive_handle(border.sobel))

def tr_border_sobel(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = border.sobel(image, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_SOBEL, {}, {'padding_str': padding_str}, channels_tr))

TR_BORDER_LAPLACIAN: str = 'laplacian'
@render_error
def build_border_laplacian_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_LAPLACIAN):
        build_tr_name_input(TR_BORDER_LAPLACIAN, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_value_int_selector('crossing threshold', 0, MAX_COLOR, default_value=100)
        build_tr_dialog_end_buttons(TR_BORDER_LAPLACIAN, image_name, tr_border_laplacian, generic_tr_inductive_handle(border.laplace))

def tr_border_laplacian(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image               = img_repo.get_image(image_name)
    new_name            = get_tr_name_value(image)
    padding_str         = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    crossing_threshold  = get_tr_int_value()
    # 2. Procesamos
    new_data, channels_tr = border.laplace(image, crossing_threshold, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_LAPLACIAN, {'crossing_threshold': crossing_threshold}, {'padding_str': padding_str}, channels_tr))

TR_BORDER_LOG: str = 'LOG'
@render_error
def build_border_log_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_LOG):
        build_tr_name_input(TR_BORDER_LOG, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_value_float_selector('sigma', 1, 20, default_value=2)
        build_tr_value_int_selector('crossing threshold', 0, MAX_COLOR, default_value=10)
        build_tr_dialog_end_buttons(TR_BORDER_LOG, image_name, tr_border_log, generic_tr_inductive_handle(border.log))

def tr_border_log(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    sigma = get_tr_float_value()
    crossing_threshold = get_tr_int_value()
    # 2. Procesamos
    new_data, channels_tr = border.log(image, sigma, crossing_threshold, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_LOG, {'sigma': sigma, 'crossing_threshold': crossing_threshold}, {'padding_str': padding_str}, channels_tr))

TR_BORDER_SUSAN: str = 'susan'
@render_error
def build_border_susan_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_SUSAN):
        build_tr_name_input(TR_BORDER_SUSAN, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_BORDER_SUSAN, image_name, tr_border_susan, generic_tr_inductive_handle(border.susan))

def tr_border_susan(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    # 2. Procesamos
    new_data, channels_tr = border.susan(image, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_SUSAN, {}, {'padding_str': padding_str}, channels_tr))

TR_BORDER_HOUGH_LINE: str = 'hough_line'
@render_error
def build_border_hough_line_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_HOUGH_LINE):
        build_tr_name_input(TR_BORDER_HOUGH_LINE, image_name)
        build_tr_value_float_selector('t', 0, 20, default_value=1, tag='threshold')
        build_tr_value_float_selector('ratio', 0, 1, default_value=0.8, tag='ratio')
        build_tr_value_int_list_selector('theta', -90, 90, default_value=0, tag='theta')
        build_tr_value_range_selector('rho', min_val=0, max_val=255, max_count=256, tag='rho')  # TODO(nacho): mejores defaults
        build_tr_dialog_end_buttons(TR_BORDER_HOUGH_LINE, image_name, tr_border_hough_line, generic_tr_inductive_handle(border.hough_lines))

def tr_border_hough_line(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    threshold   = get_tr_float_value('threshold')
    ratio       = get_tr_float_value('ratio')
    theta       = get_tr_int_list_value('theta')
    rho         = get_tr_range_value('rho')
    # 2. Procesamos
    new_data, channels_tr = border.hough_lines(image, theta, rho, threshold, ratio)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_HOUGH_LINE, {'threshold': threshold, 'most_fitted_ratio': ratio, 'theta': theta, 'rho': rho}, {}, channels_tr))

TR_BORDER_HOUGH_CIRCLE: str = 'hough_circle'
@render_error
def build_border_hough_circle_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_HOUGH_CIRCLE):
        build_tr_name_input(TR_BORDER_HOUGH_CIRCLE, image_name)
        build_tr_value_float_selector('t', 0, 20, default_value=1.5, tag='threshold')
        build_tr_value_float_selector('ratio', 0, 1, default_value=0.6, tag='ratio')
        build_tr_value_range_selector('radius', min_val=0, max_val=200, max_count=1, tag='radius')
        build_tr_value_range_selector('x', min_val=0, max_val=239, max_count=120, tag='x')
        build_tr_value_range_selector('y', min_val=0, max_val=239, max_count=120, tag='y')
        build_tr_dialog_end_buttons(TR_BORDER_HOUGH_CIRCLE, image_name, tr_border_hough_circle, generic_tr_inductive_handle(border.hough_lines))

def tr_border_hough_circle(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    threshold   = get_tr_float_value('threshold')
    ratio       = get_tr_float_value('ratio')
    radius      = get_tr_range_value('radius')
    x           = get_tr_range_value('x')
    y           = get_tr_range_value('y')
    # 2. Procesamos
    new_data, channels_tr = border.hough_circles(image, radius, x, y, threshold, ratio)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_HOUGH_CIRCLE, {'threshold': threshold, 'most_fitted_ratio': ratio, 'radius': radius, 'x': x, 'y': y}, {}, channels_tr))

TR_BORDER_CANNY: str = 'canny'
@render_error
def build_border_canny_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_LOG):
        build_tr_name_input(TR_BORDER_CANNY, image_name)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_value_int_selector('lower threshold', 0, MAX_COLOR, default_value=10, tag='t1')
        build_tr_value_int_selector('upper threshold', 0, MAX_COLOR, default_value=70, tag='t2')
        build_tr_dialog_end_buttons(TR_BORDER_CANNY, image_name, tr_border_canny, generic_tr_inductive_handle(border.canny))

def tr_border_canny(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    t1 = get_tr_int_value('t1')
    t2 = get_tr_int_value('t2')
    # 2. Procesamos
    new_data, channels_tr = border.canny(image, t1, t2, padding_str)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_CANNY, {'t1': t1, 't2': t2}, {'padding_str': padding_str}, channels_tr))

TR_BORDER_HARRIS: str = 'harris'
@render_error
def build_border_harris_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_HARRIS):
        build_tr_name_input(TR_BORDER_HARRIS, image_name)
        build_tr_value_int_selector('sigma', 1, 10, default_value=2, tag='sigma')
        build_tr_value_float_selector('k', 0.01, 0.1, default_value=0.04, tag='k')
        build_tr_value_float_selector('threshold', 20, 100.0, default_value=40, tag='threshold')
        build_tr_radio_buttons(border.HarrisR.names(), tag='function')
        build_tr_checkbox('Include Borders', default_value=True)
        build_tr_radio_buttons(PaddingStrategy.names())
        build_tr_dialog_end_buttons(TR_BORDER_HARRIS, image_name, tr_border_harris, generic_tr_inductive_handle(border.harris))

def tr_border_harris(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    padding_str = PaddingStrategy.from_str(get_tr_radio_buttons_value())
    sigma       = get_tr_int_value('sigma')
    k           = get_tr_float_value('k')
    threshold   = get_tr_float_value('threshold')
    function    = border.HarrisR.from_str(get_tr_radio_buttons_value('function'))
    with_border = get_tr_checkbox_value()
    # 2. Procesamos
    new_data, channels_tr = border.harris(image, sigma, k, threshold, function, padding_str, with_border)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_HARRIS, {'sigma': sigma, 'k': k, 'threshold':threshold, 'r_function':function}, {'padding_str': padding_str, 'with_border': with_border}, channels_tr))

TR_BORDER_ACTIVE_OUTLINE: str = 'active_outline'
@render_error
def build_border_active_outline_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_ACTIVE_OUTLINE):
        build_tr_name_input(TR_BORDER_ACTIVE_OUTLINE, image_name)
        build_tr_dialog_end_buttons(TR_BORDER_ACTIVE_OUTLINE, image_name, tr_border_active_outline_base, tr_border_active_outline_inductive)

def tr_border_active_outline_base(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image           = img_repo.get_image(image_name)
    new_name        = get_tr_name_value(image)
    rect_selection  = interface.get_image_window_rect_selections(f'image_window_{image_name}')
    if rect_selection is None:
        raise ValueError('An initial bounding box on first frame is required in Active Outline')
    # 2. Procesamos
    new_data, channels_tr = border.active_outline_base(image, rect_selection)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_ACTIVE_OUTLINE, {'rect_selection': rect_selection}, {}, channels_tr))

def tr_border_active_outline_inductive(new_name: str, frame: int, prev: Image, current: Image) -> Image:
    new_data, channels_tr = border.active_outline_inductive(frame, prev, current)
    return current.transform(new_name, new_data, ImageTransformation(TR_BORDER_ACTIVE_OUTLINE, {}, {}, channels_tr))

TR_BORDER_MULTIPLE_ACTIVE_OUTLINE: str = 'multiple_active_outline'
@render_error
def build_border_multiple_active_outline_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_BORDER_MULTIPLE_ACTIVE_OUTLINE):
        build_tr_name_input(TR_BORDER_MULTIPLE_ACTIVE_OUTLINE, image_name)
        build_tr_value_float_selector('threshold', 0.01, 1, default_value=0.5, tag='threshold')
        build_tr_dialog_end_buttons(TR_BORDER_MULTIPLE_ACTIVE_OUTLINE, image_name, tr_border_multiple_active_outline_base, tr_border_multiple_active_outline_inductive)

def tr_border_multiple_active_outline_base(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image           = img_repo.get_image(image_name)
    threshold       = get_tr_float_value('threshold')
    new_name        = get_tr_name_value(image)
    rect_selection  = interface.get_image_window_rect_selections(f'image_window_{image_name}')
    if rect_selection is None:
        raise ValueError('An initial bounding box on first frame is required in Active Outline')
    # 2. Procesamos
    new_data, channels_tr = border.multiple_active_outline_base(image, rect_selection, threshold)
    # 3. Creamos Imagen
    return image.transform(new_name, new_data, ImageTransformation(TR_BORDER_MULTIPLE_ACTIVE_OUTLINE, {'rect_selection': rect_selection, 'threshold': threshold}, {}, channels_tr))

def tr_border_multiple_active_outline_inductive(new_name: str, frame: int, prev: Image, current: Image) -> Image:
    new_data, channels_tr = border.active_outline_inductive(frame, prev, current)
    return current.transform(new_name, new_data, ImageTransformation(TR_BORDER_MULTIPLE_ACTIVE_OUTLINE, {}, {}, channels_tr))

########################################################
# ********************* Combine ********************** #
########################################################

TR_COMBINE_ADD: str = 'add'
@render_error
def build_combine_add_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COMBINE_ADD):
        build_tr_name_input(TR_COMBINE_ADD, image_name)
        build_tr_img_selector(image_name, same_shape=True)
        build_tr_dialog_end_buttons(TR_COMBINE_ADD, image_name, tr_combine_add, generic_tr_inductive_handle(combine.add))

def tr_combine_add(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sec_image   = get_tr_img_value()
    require_same_shape(image, sec_image, 'You can only sum images with the same shape')
    # 2. Procesamos
    new_data, channels_tr = combine.add(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return image.transform(new_name, new_data, ImageTransformation(TR_COMBINE_ADD, {'sec_image_name': sec_image.name}, {'second_image': sec_image}, channels_tr))

TR_COMBINE_SUB: str = 'sub'
@render_error
def build_combine_sub_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COMBINE_SUB):
        build_tr_name_input(TR_COMBINE_SUB, image_name)
        build_tr_img_selector(image_name, same_shape=True)
        build_tr_dialog_end_buttons(TR_COMBINE_SUB, image_name, tr_combine_sub, generic_tr_inductive_handle(combine.sub))

def tr_combine_sub(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sec_image   = get_tr_img_value()
    require_same_shape(image, sec_image, 'You can only sub images with the same shape')
    # 2. Procesamos
    new_data, channels_tr = combine.sub(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return image.transform(new_name, new_data, ImageTransformation(TR_COMBINE_SUB, {'sec_image_name': sec_image.name}, {'second_image': sec_image}, channels_tr))

TR_COMBINE_MULT: str = 'multiply'
@render_error
def build_combine_mult_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COMBINE_MULT):
        build_tr_name_input(TR_COMBINE_MULT, image_name)
        build_tr_img_selector(image_name, same_shape=True)
        build_tr_dialog_end_buttons(TR_COMBINE_MULT, image_name, tr_combine_mult, generic_tr_inductive_handle(combine.multiply))

def tr_combine_mult(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sec_image   = get_tr_img_value()
    require_same_shape(image, sec_image, 'You can only multiply images with the same shape')
    # 2. Procesamos
    new_data, channels_tr = combine.multiply(image, sec_image)
    # 3. Creamos Imagen y finalizamos
    return image.transform(new_name, new_data, ImageTransformation(TR_COMBINE_MULT, {'sec_image_name': sec_image.name}, {'sec_image': sec_image}, channels_tr))

TR_COMBINE_SIFT: str = 'sift'
@render_error
def build_combine_sift_dialog(image_name: str) -> None:
    with build_tr_dialog(TR_COMBINE_SIFT):
        build_tr_name_input(TR_COMBINE_SIFT, image_name)
        build_tr_img_selector(image_name)
        build_tr_value_int_selector('Features',             0, 20, default_value=0, tag='features')
        build_tr_value_int_selector('Layers',               0, 20, default_value=3, tag='layers')
        build_tr_value_float_selector('Contrast Threshold', 0, 1, default_value=0.1, tag='contrast_t')
        build_tr_value_float_selector('Edge Threshold',     0, 100, default_value=10, tag='edge_t')
        build_tr_value_float_selector('Gauss Sigma',        0, 100, default_value=1.6, tag='sigma')
        build_tr_value_float_selector('Match Threshold',    0, 1000, default_value=200, tag='match_t')
        build_tr_checkbox('cross_check', default_value=True)
        build_tr_dialog_end_buttons(TR_COMBINE_SIFT, image_name, tr_combine_sift, generic_tr_inductive_handle(combine.sift))

def tr_combine_sift(image_name: str) -> Image:
    # 1. Obtenemos inputs
    image       = img_repo.get_image(image_name)
    new_name    = get_tr_name_value(image)
    sec_image   = get_tr_img_value()
    features    = get_tr_int_value('features')
    layers      = get_tr_int_value('layers')
    contrast_t  = get_tr_float_value('contrast_t')
    edge_t      = get_tr_float_value('edge_t')
    sigma       = get_tr_float_value('sigma')
    match_t     = get_tr_float_value('match_t')
    cross_check = get_tr_checkbox_value()
    # 2. Procesamos
    new_data, channels_tr = combine.sift(image, sec_image, features, layers, contrast_t, edge_t, sigma, match_t, cross_check)
    # 3. Creamos Imagen y finalizamos
    return image.transform(new_name, new_data, ImageTransformation(
        TR_COMBINE_SIFT,
       {'img2_name': sec_image.name, 'features': features, 'layers': layers, 'contrast_t': contrast_t, 'edge_t': edge_t, 'sigma': sigma, 'match_t': match_t, 'cross_check': cross_check},
       {'img2': sec_image},
       channels_tr
    ))
