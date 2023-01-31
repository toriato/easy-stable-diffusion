from IPython.display import display
from ipywidgets import widgets

from modules import shared
from modules.log import Log
from modules.utils import mount_google_drive

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
    if shared.IN_COLAB:
        with log:
            mount_google_drive()

    def on_click(_):
        with log:
            # 실행할 때 필요한 패키지들이 import 되자마자 실행되기 때문에 초기 실행기 느려질 수 있음
            # 그러므로 사용자가 작업을 실행할 때 하위 모듈을 가져와야함
            from workflows.stable_diffusion_webui_modules.launch import launch
            launch()

    button = widgets.Button(
        description='실행',
        layout={'width': 'calc(100% - 1em)'}
    )

    button.on_click(on_click)

    # 인터페이스에 사용자 설정 추가하기
    from workflows.stable_diffusion_webui_modules.control import grids
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
