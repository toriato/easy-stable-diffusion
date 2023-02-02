from pathlib import Path
from typing import List, NamedTuple


class Options(NamedTuple):
    workspace: Path
    disconnect_runtime: bool
    use_google_drive: bool
    use_xformers: bool
    use_gradio: bool
    gradio_username: str
    gradio_password: str
    ngrok_api_token: str
    repo_url: str
    repo_commit: str
    args: List[str]
    extra_args: List[str]


options: Options
