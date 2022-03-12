import dearpygui.dearpygui as dpg

from interface import build_load_image_dialog, build_load_metadata_dialog, create_circle_handler, create_square_handler, \
    LOAD_IMAGE_DIALOG, LOAD_METADATA_DIALOG, PRIMARY_WINDOW, TEXTURE_REGISTRY, IMAGES_MENU, build_save_image_dialog, \
    build_image_handler_registry, build_hist_themes
from metadata_repo import set_metadata_file
DEFAULT_METADATA_PATH: str = 'images/raw_metadata.tsv'

def main():
    set_metadata_file(DEFAULT_METADATA_PATH)

    dpg.create_context()
    dpg.create_viewport(title='Analisis y Tratamiento de Imagenes')  # TODO: Add icons
    dpg.setup_dearpygui()

    # Registries
    dpg.add_texture_registry(tag=TEXTURE_REGISTRY)
    build_image_handler_registry()

    # Dialog
    build_load_image_dialog()
    build_save_image_dialog()
    build_load_metadata_dialog()
    
    # Image window
    build_hist_themes()

    with dpg.viewport_menu_bar():
        dpg.add_menu_item(label='Load', callback=lambda: dpg.show_item(LOAD_IMAGE_DIALOG))

        with dpg.menu(label='Create'):
            dpg.add_menu_item(label='Circle', callback=create_circle_handler)
            dpg.add_menu_item(label='Square', callback=create_square_handler)

        dpg.add_menu(label='Catalog', tag=IMAGES_MENU)

        with dpg.menu(label='Configuration'):
            dpg.add_text('Metadata file for raw images: ')
            dpg.add_menu_item(label='Load Configuration file', callback=lambda: dpg.show_item(LOAD_METADATA_DIALOG))

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
