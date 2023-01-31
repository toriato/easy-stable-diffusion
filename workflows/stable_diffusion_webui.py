from ipywidgets import widgets
from IPython.display import display

from modules.log import Log
from modules.workspace import mount_google_drive

mount_google_drive()

log = Log()
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


def main():
    with log:
        from workflows.stable_diffusion_webui_modules import grids, context, launch

    button = widgets.Button(
        description='실행',
        layout={'width': 'calc(100% - 1em)'}
    )

    def on_click(_):
        with log:
            launch(context)

    button.on_click(on_click)

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
