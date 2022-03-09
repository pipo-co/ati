import dearpygui.dearpygui as dpg

from image_utils import raw_images_metadata_path
from interface import build_load_image_dialog, LOAD_IMAGE_DIALOG, PRIMARY_WINDOW

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
