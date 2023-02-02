import sys
from pathlib import Path

file_path = Path(__file__)
sys.path.append(str(file_path.parent))
sys.path.append(str(file_path.parent.parent))


def main(**kwargs):
    from modules import shared
    from modules.log import Log

    log = Log(only_last_lines=0)

    if shared.IN_INTERACTIVE:
        from IPython.display import display
        display(log.widget)

    with log:
        from stable_diffusion_webui_modules import setup
        setup.setup_options(**kwargs)
        setup.setup_repository()
        setup.setup_dependencies()


if __name__ == '__main__':
    main()
