from modules.log import Log

log = Log()

try:
    from IPython.display import display
    display(log.widget)
except ImportError:
    pass


def main(**kwargs):
    with log:
        from .stable_diffusion_webui_modules.environment import setup_options
        setup_options(**kwargs)
