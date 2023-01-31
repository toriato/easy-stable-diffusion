from ipywidgets import widgets
from IPython.display import display

from modules.workspace import mount_google_drive

from workflows.stable_diffusion_webui_modules import *


def main():
    mount_google_drive()

    controls = widgets.VBox()
    wrapper = widgets.GridBox(
        (controls, log.widget),
        layout={
            'width': '100%',
            'padding': '.5em',
            'grid_template_columns': '1fr 1fr',
            'grid_gap': '.5em'
        }
    )

    display(wrapper)

    button = widgets.Button(
        description='실행',
        layout={'width': 'calc(100% - 1em)'}
    )

    button.on_click(lambda _: launch(context))

    controls.children = (
        *[
            widgets.GridBox([
                control.wrapper
                for control in grid
                if control.wrapper
            ], layout={
                'width': '100%',
                'padding': '.5em',
                'grid_template_columns': 'repeat(auto-fit, minmax(300px, 1fr))',
                'grid_gap': '1em',
                **grid.layout
            })
            for grid in grids
        ],
        button
    )
