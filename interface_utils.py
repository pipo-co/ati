import functools

import dearpygui.dearpygui as dpg

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
