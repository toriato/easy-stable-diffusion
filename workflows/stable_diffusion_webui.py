from pathlib import Path
from ipywidgets import widgets
from IPython.display import display

from modules.alert import alert
from modules.log import Log
from modules.ui import Selector, SelectorDownloader, FormSet, Input
from modules.shared import workspace, workspace_lookup_generator

log = Log(
    Path('temp.log'),
    widget=widgets.HTML()
)
controls = widgets.VBox()
wrapper = widgets.GridBox(
    (controls, log.widget),
    layout={
        'padding': '.5em',
        'grid_template_columns': '2fr 1fr',
        'grid_gap': '.5em'
    }
)

display(wrapper)

try:
    ckpt = Selector(
        options=[
            SelectorDownloader()
        ],
        refresher=workspace_lookup_generator([
            'models/Stable-diffusion/**/*.ckpt',
            'models/Stable-diffusion/**/*.safetensors'
        ])
    )

    vae = Selector(
        options=[
            SelectorDownloader()
        ],
        refresher=workspace_lookup_generator([
            'models/VAE/**/*.pt',
            'models/VAE/**/*.ckpt',
            'models/VAE/**/*.safetensors'
        ])
    )

    formsets = [
        FormSet(
            Input(
                'workspace',
                workspace.create_ui(),
                '''
                <h3>데이터 경로</h3>
                <p>모델, 출력 파일 등을 보관할 경로</p>
                '''
            )
        ),
        FormSet(
            Input(
                '--ckpt',
                ckpt.create_ui(),
                '''
                <h3>체크포인트 경로</h3>
                '''
            ),
            Input(
                '--vae_path',
                vae.create_ui(),
                '''
                <h3>VAE 경로</h3>
                '''
            )
        )
    ]

    controls.children = (
        *[
            widgets.GridBox(
                ([i.create_ui() for i in formset]),
                layout={
                    'width': '100%',
                    'padding': '.5em',
                    'grid_template_columns': 'repeat(auto-fit, minmax(400px, 1fr))',
                    'grid_gap': '1em',
                    **formset.layout
                }
            )
            for formset in formsets
        ],
    )

except Exception as e:
    alert(f'초기화 중 오류가 발생했습니다\n{e}')
    raise
