from modules.log import Log

log = Log(only_last_lines=0)

try:
    from IPython.display import display
    display(log.widget)
except ImportError:
    pass


def main(**kwargs):
    with log:
        from .stable_diffusion_webui_modules import environment
        environment.setup_options(**kwargs)
        environment.setup_repository()
        environment.setup_dependencies()
