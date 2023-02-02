from modules.log import Log

log = Log(only_last_lines=0)

try:
    from IPython.display import display
    display(log.widget)
except ImportError:
    pass


def main(**kwargs):
    with log:
        from .stable_diffusion_webui_modules import setup
        setup.setup_options(**kwargs)
        setup.setup_repository()
        setup.setup_dependencies()


if __name__ == '__main__':
    main()
