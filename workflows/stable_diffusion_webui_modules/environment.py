from pathlib import Path
from typing import List, NamedTuple


class Options(NamedTuple):
    workspace: Path = Path()
    disconnect_runtime: bool = True
    use_google_drive: bool = True
    use_xformers: bool = True
    use_gradio: bool = False
    gradio_username: str = ''
    gradio_password: str = ''
    ngrok_api_token: str = ''
    python_executable: str = ''
    repo_url: str = ''
    repo_commit: str = ''
    args: List[str] = []
    extra_args: List[str] = []


options: Options
options_ignore_override = [
    'workspace'
]
