import itertools

from modules.workspace import workspace, workspace_lookup_generator
from modules.control import Control, ControlContext, ControlGrid
from modules.ui import Option, Text, Selector

grids = [
    ControlGrid([
        Control(
            workspace,
            summary_html='''
                <h3>데이터 디렉터리 경로</h3>
                <p>모델이나 결과 이미지 등을 불러오거나 저장할 디렉터리 경로</p>
                '''
        ),
    ]),
    ControlGrid([
        Control(
            Selector([
                Option('< 기본 버전 사용 >', lambda _: 'python'),
                Option('Python 3.10', lambda _: 'python3.10'),
                Option('Python 3.9', lambda _: 'python3.9'),
                Option('Python 3.8', lambda _: 'python3.8'),
                Option('Python 3.7', lambda _: 'python3.7'),
            ]),
            key='python_executable',
            summary_html='<h3>Python 버전</h3>'
        ),
        Control(
            Text('https://github.com/AUTOMATIC1111/stable-diffusion-webui.git'),
            key='repository',
            summary_html='<h3>레포지토리 경로</h3>'
        ),
        Control(
            Text(),
            key='repository_commit',
            summary_html='<h3>레포지토리 커밋</h3>'
        )
    ], layout={
        'grid_template_columns': 'repeat(auto-fit, minmax(250px, 1fr))'
    }),

    ControlGrid([
        Control(
            Selector(
                [
                    Text(
                        name='< 주소로부터 파일 받아오기 >',
                        default_text=''
                    )
                ],
                workspace_lookup_generator([
                    'models/Stable-diffusion/**/*.ckpt',
                    'models/Stable-diffusion/**/*.safetensors'
                ])
            ),
            argument='--ckpt',
            summary_html='<h3>체크포인트 경로</h3>'
        ),
        Control(
            Selector(
                [
                    Text(
                        name='< 주소로부터 파일 받아오기 >',
                        default_text=''
                    )
                ],
                workspace_lookup_generator([
                    'models/VAE/**/*.pt',
                    'models/VAE/**/*.ckpt',
                    'models/VAE/**/*.safetensors'
                ])
            ),
            argument='--vae_path',
            summary_html='<h3>VAE 경로</h3>'
        )
    ])
]

context: ControlContext = {
    control.key: control
    for control in itertools.chain(*grids)
    if control.key
}
