from ipywidgets import widgets
from IPython.display import display

from modules import shared
from modules.workspace import workspace, workspace_lookup_generator
from modules.alert import alert
from modules.log import Log
from modules.ui import Selector, SelectorText, Input, InputSet

if shared.IN_COLAB:
    try:
        # 마운트 후 발생하는 출력을 제거하기 위해 새 위젯 컨텍스 만들기
        output = widgets.Output()

        with output:
            from google.colab import drive
            drive.mount(shared.GDRIVE_MOUNT_DIR)
            output.clear_output()

    except ImportError:
        alert('구글 드라이브에 접근할 수 없습니다, 동기화를 사용할 수 없습니다!')

log: Log
log_html = widgets.HTML

controls = widgets.VBox()
wrapper = widgets.GridBox(
    (controls, log_html),
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
            SelectorText()
        ],
        refresher=workspace_lookup_generator([
            'models/Stable-diffusion/**/*.ckpt',
            'models/Stable-diffusion/**/*.safetensors'
        ])
    )

    vae = Selector(
        options=[
            SelectorText()
        ],
        refresher=workspace_lookup_generator([
            'models/VAE/**/*.pt',
            'models/VAE/**/*.ckpt',
            'models/VAE/**/*.safetensors'
        ])
    )

    formsets = [
        InputSet(
            Input(
                'workspace',
                workspace.create_ui(),
                '''
                <h3>데이터 경로</h3>
                <p>모델, 출력 파일 등을 보관할 경로</p>
                '''
            )
        ),
        InputSet(
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
