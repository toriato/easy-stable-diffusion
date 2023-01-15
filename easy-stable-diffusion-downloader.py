import shlex

from os import PathLike
from typing import Union, List, Dict
from pathlib import Path
from IPython.display import display
from ipywidgets import widgets
from google.colab import drive, runtime

# fmt: off
#@title

#@markdown ### <font color="orange">***작업 디렉터리 경로***</font>
#@markdown 모델 파일 등이 영구적으로 보관될 디렉터리 경로
WORKSPACE = 'SD' #@param {type:"string"}

#@markdown ##### <font color="orange">***다운로드가 끝나면 자동으로 코랩 런타임을 종료할지?***</font>
DISCONNECT_RUNTIME = True  #@param {type:"boolean"}

# fmt: on

# 인터페이스 요소
dropdowns = widgets.VBox()
output = widgets.Output()
download_button = widgets.Button(
    description='다운로드',
    disabled=True,
    layout={"width": "99%"}
)

display(
    widgets.HBox(children=(
        widgets.VBox(
            children=(dropdowns, download_button),
            layout={"margin-right": "1em"}
        ),
        output
    )))


# 파일 경로
workspace_dir = Path('drive', 'MyDrive', WORKSPACE)
sd_model_dir = workspace_dir.joinpath('models', 'Stable-diffusion')
sd_embedding_dir = workspace_dir.joinpath('embeddings')
vae_dir = workspace_dir.joinpath('models', 'VAE')

# 구글 드라이브 마운팅
with output:
    drive.mount('drive')

sd_model_dir.mkdir(0o777, True, True)
sd_embedding_dir.mkdir(0o777, True, True)
vae_dir.mkdir(0o777, True, True)


class File:
    prefix: Path

    def __init__(self, url: str, path: PathLike = None, *extra_args: List[str]) -> None:
        if self.prefix:
            if not path:
                path = self.prefix
            elif type(path) == str:
                path = self.prefix.joinpath(path)

        self.url = url
        self.path = Path(path)
        self.extra_args = extra_args

    def download(self) -> None:
        output.clear_output()
        args = shlex.join((
            '--continue',
            '--always-resume',
            '--summary-interval', '3',
            '--console-log-level', 'error',
            '--max-concurrent-downloads', '16',
            '--max-connection-per-server', '16',
            '--split', '16',
            '--dir' if self.path.is_dir() else '--out', str(self.path.name),
            *self.extra_args,
            self.url
        ))

        with output:
            # aria2 로 파일 받아오기
            # fmt: off
            !which aria2c || apt install -y aria2
            output.clear_output()

            print('aria2 를 사용해 파일을 받아옵니다.')
            !aria2c {args}
            output.clear_output()

            print('파일을 성공적으로 받았습니다, 드라이브로 이동합니다.')
            print('이 작업은 파일의 크기에 따라 5분 이상 걸릴 수도 있으니 잠시만 기다려주세요.')
            if DISCONNECT_RUNTIME:
                print('작업이 완료되면 런타임을 자동으로 해제하니 다른 작업을 진행하셔도 좋습니다.')

            !rsync --remove-source-files -aP "{str(self.path.name)}" "{str(self.path)}"

            # fmt: on
            print('파일을 성공적으로 옮겼습니다, 이제 런타임을 해제해도 좋습니다.')


class ModelFile(File):
    prefix = sd_model_dir


class EmbeddingFile(File):
    prefix = sd_embedding_dir


class VaeFile(File):
    prefix = vae_dir


# 모델 목록
files = {
    'Stable-Diffusion Checkpoints': {
        # 현재 목록의 키 값 정렬해서 보여주기
        '$sort': True,

        'Stable Diffusion': {
            'v2.1': {
                '768-v': {
                    'ema-pruned': {
                        'safetensors': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors',
                            'stable-diffusion-v2-1-786-v-ema-pruned.safetensors'),
                        'ckpt': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt',
                            'stable-diffusion-v2-1-786-v-ema-pruned.ckpt'),
                    },
                    'nonema-pruned': {
                        'safetensors': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.safetensors',
                            'stable-diffusion-v2-1-786-v-nonema-pruned.safetensors'),
                        'ckpt': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.ckpt',
                            'stable-diffusion-v2-1-786-v-nonema-pruned.ckpt'),
                    }
                },
                '512-base': {
                    'ema-pruned': {
                        'safetensors': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors',
                            'stable-diffusion-v2-1-512-base-ema-pruned.safetensors'),
                        'ckpt': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt',
                            'stable-diffusion-v2-1-512-base-ema-pruned.ckpt'),
                    },
                    'nonema-pruned': {
                        'safetensors': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-nonema-pruned.safetensors',
                            'stable-diffusion-v2-1-512-base-nonema-pruned.safetensors'),
                        'ckpt': ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-nonema-pruned.ckpt',
                            'stable-diffusion-v2-1-512-base-nonema-pruned.ckpt'),
                    },
                },
            },
            'v2.0': {
                '768-v-ema': {
                    'safetensors': ModelFile(
                        'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.safetensors',
                        'stable-diffusion-v2-0-786-v-ema.safetensors'),
                    'ckpt': ModelFile(
                        'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt',
                        'stable-diffusion-v2-0-786-v-ema.ckpt'),
                },
                '512-base-ema': {
                    'safetensors': ModelFile(
                        'https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.safetensors',
                        'stable-diffusion-v2-0-512-base-ema.safetensors'),
                    'ckpt': ModelFile(
                        'https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt',
                        'stable-diffusion-v2-0-512-base-ema.ckpt'),
                },
            },
            'v1.5': {
                'pruned-emaonly': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
                        'stable-diffusion-v1-5-pruned-emaonly.ckpt')
                },
                'pruned': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
                        'stable-diffusion-v1-5-pruned.ckpt')
                },
            },
        },

        'Dreamlike': {
            'photoreal': {
                'v2.0': {
                    'safetensors': ModelFile('https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.safetensors'),
                    'ckpt': ModelFile('https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.ckpt')
                },
                'v1.0': {
                    'ckpt': ModelFile('https://huggingface.co/dreamlike-art/dreamlike-photoreal-1.0/resolve/main/dreamlike-photoreal-1.0.ckpt')
                },
            },
            'diffusion': {
                'v1.0': {
                    'safetensors': ModelFile('https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0/resolve/main/dreamlike-diffusion-1.0.safetensors'),
                    'ckpt': ModelFile('https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0/resolve/main/dreamlike-diffusion-1.0.ckpt')
                },
            }
        },

        'Waifu Diffusion': {
            'v1.4': {
                'anime_e1': {
                    'ckpt': [
                        ModelFile(
                            'https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-1-4-anime_e1.ckpt',
                            'waifu-diffusion-v1-4-anime-e1.ckpt'),
                        ModelFile(
                            'https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-1-4-anime_e1.yaml',
                            'waifu-diffusion-v1-4-anime-e1.yaml'),
                    ]
                },
                'booru-step-14000-unofficial': {
                    'safetensors': ModelFile(
                        'https://huggingface.co/waifu-diffusion/unofficial-releases/resolve/main/wd14-booru-step-14000-unofficial.safetensors',
                        'waifu-diffusion-v1-4-booru-step-14000.ckpt'),
                },
            },
            'v1.3.5': {
                '80000-fp32': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/wd-1-3-5_80000-fp32.ckpt',
                        'waifu-diffusion-v1-4-80000-fp32.ckpt'),
                },
                'penultimate-ucg-cont': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/wd-1-3-penultimate-ucg-cont.ckpt',
                        'waifu-diffusion-v1-3-5-penultimate-ucg-cont.ckpt'),
                }
            },
            'v1.3': {
                'fp16': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt',
                        'waifu-diffusion-v1-3-float16.ckpt')
                },
                'fp32': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float32.ckpt',
                        'waifu-diffusion-v1-3-float32.ckpt')
                },
                'full': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-full.ckpt',
                        'waifu-diffusion-v1-3-full.ckpt')
                },
                'full-opt': {
                    'ckpt': ModelFile(
                        'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-full-opt.ckpt',
                        'waifu-diffusion-v1-3-full-opt.ckpt')
                },
            },
        },

        'TrinArt': {
            'derrida_characters': {
                'v2': {
                    'final': {
                        'ckpt': ModelFile(
                            'https://huggingface.co/naclbit/trinart_derrida_characters_v2_stable_diffusion/resolve/main/derrida_final.ckpt',
                            'trinart_characters_v2_final.ckpt')
                    },
                },
                'v1 (19.2m)': {
                    'ckpt': ModelFile('https://huggingface.co/naclbit/trinart_characters_19.2m_stable_diffusion_v1/resolve/main/trinart_characters_it4_v1.ckpt')
                },
            },
            'v2': {
                '115000': {
                    'ckpt': ModelFile('https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step115000.ckpt'),
                },
                '95000': {
                    'ckpt': ModelFile('https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step95000.ckpt'),
                },
                '60000': {
                    'ckpt': ModelFile('https://huggingface.co/naclbit/trinart_stable_diffusion_v2/resolve/main/trinart2_step60000.ckpt'),
                },
            },
        },

        'OrangeMixs': {
            'AbyssOrangeMix': {
                '2': {
                    'first-party': {
                        'hard': {
                            'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix2/AbyssOrangeMix2_hard.safetensors'),
                        },
                        'nsfw': {
                            'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix2/AbyssOrangeMix2_nsfw.safetensors'),
                        },
                        'sfw': {
                            'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix2/AbyssOrangeMix2_sfw.safetensors'),
                            'ckpt': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix2/AbyssOrangeMix2_sfw.ckpt')
                        }
                    },
                    'nonema-pruned': {
                        'hard': {
                            'safetensors': ModelFile('https://huggingface.co/Kaeya/aichan_blend/resolve/main/AbyssOrangeMix2_sfw-pruned.safetensors'),
                        },
                        'nsfw': {
                            'safetensors': ModelFile('https://huggingface.co/Kaeya/aichan_blend/resolve/main/AbyssOrangeMix2_nsfw-pruned.safetensors'),
                        },
                        'sfw': {
                            'safetensors': ModelFile('https://huggingface.co/Kaeya/aichan_blend/resolve/main/AbyssOrangeMix2_hard-pruned.safetensors'),
                        }
                    }
                },
                '1': {
                    'half': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix/AbyssOrangeMix_half.safetensors'),
                    },
                    'night': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix/AbyssOrangeMix_Night.safetensors'),
                    },
                    'base': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix/AbyssOrangeMix_base.ckpt'),
                    },
                }
            },
            'EerieOrangeMix': {
                '2': {
                    'half': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix2_half.safetensors'),
                    },
                    'night': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix2_night.safetensors'),
                    },
                    'base': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix2.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix2_base.ckpt'),
                    }
                },
                '1': {
                    'half': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix_half.safetensors'),
                    },
                    'night': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix_night.safetensors'),
                    },
                    'base': {
                        'safetensors': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/EerieOrangeMix/EerieOrangeMix_base.ckpt'),
                    }
                },
            },
        },

        'Anything': {
            'v4.5 (unofficial merge)': {
                'safetensors': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.5-pruned.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.5-pruned.ckpt'),
            },
            'v4.0 (unofficial merge)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp16.ckpt'),
                    },
                    'fp32': {
                        'safetensors': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp32.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp32.ckpt'),
                    },
                    'safetensors': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned.safetensors'),
                    'ckpt': ModelFile('https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned.ckpt'),
                }
            },
            'v3.0': {
                'better-vae': {
                    'fp32': {
                        'safetensors': ModelFile(
                            'https://huggingface.co/Linaqruf/anything-v3-better-vae/resolve/main/any-v3-fp32-better-vae.safetensors',
                            'anything-v3-0-fp32-better-vae.safetensors'),
                        'ckpt': ModelFile(
                            'https://huggingface.co/Linaqruf/anything-v3-better-vae/resolve/main/any-v3-fp32-better-vae.ckpt',
                            'anything-v3-0-fp32-better-vae.ckpt'),
                    }
                },
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile(
                            'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned-fp16.safetensors',
                            'anything-v3-0-pruned-fp16.safetensors'),
                        'ckpt': ModelFile(
                            'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned-fp16.ckpt',
                            'anything-v3-0-pruned-fp16.ckpt'),
                    },
                    'fp32': {
                        'safetensors': ModelFile(
                            'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned-fp32.safetensors',
                            'anything-v3-0-pruned-fp32.safetensors'),
                        'ckpt': ModelFile(
                            'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned-fp32.ckpt',
                            'anything-v3-0-pruned-fp32.ckpt')
                    },
                    'safetensors': ModelFile(
                        'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned.safetensors',
                        'anything-v3-0-pruned.safetensors'),
                    'ckpt': ModelFile(
                        'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned.ckpt',
                        'anything-v3-0-pruned.ckpt'),
                },
                'safetensors': ModelFile(
                    'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0.safetensors',
                    'anything-v3-0.safetensors'),
                'ckpt': ModelFile(
                    'https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0.ckpt',
                    'anything-v3-0.ckpt'),
            },
        },

        'Protogen': {
            'Dragon (RPG themes)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon.ckpt'),
            },
            'x5.8 (Sci-Fi + Anime)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8.ckpt'),
            },
            'x5.3 (Photorealism)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3.ckpt'),
            },
            'x3.4 (Photorealism)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4.ckpt'),
            },
            'x2.2 (Anime)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_v2.2_Official_Release/resolve/main/Protogen_V2.2-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_v2.2_Official_Release/resolve/main/Protogen_V2.2-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_v2.2_Official_Release/resolve/main/Protogen_V2.2.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_v2.2_Official_Release/resolve/main/Protogen_V2.2.ckpt'),
            },
        },

        '7th_Layer': {
            '7th_anime': {
                'v2.5': {
                    'B': {
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2.5_B.ckpt'),
                    },
                },
                'v2.0': {
                    'A': {
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2_A.ckpt'),
                    },
                    'B': {
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2_B.ckpt'),
                    },
                    'C': {
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2_C.ckpt'),
                    },
                },
            },
            'abyss_7th_layer': {
                'v1.1': {
                    'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/Abyss_7th_anime_v1.1.ckpt'),
                },
                'v1.0': {
                    'G1': {
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/abyss_7th_layerG1.ckpt'),
                    },
                    'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/abyss_7th_layer.ckpt'),
                }
            }
        }
    },

    'VAEs': {
        '$sort': True,

        'Stable Diffusion': {
            'vae-ft-mse-840000': {
                'pruned': {
                    'safetensors': VaeFile(
                        'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors',
                        'stable-diffusion-vae-ft-mse-840000-ema-pruned.safetensors'),
                    'ckpt': VaeFile(
                        'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt',
                        'stable-diffusion-vae-ft-mse-840000-ema-pruned.ckpt')
                }
            },
            'vae-ft-ema-560000': {
                'safetensors': VaeFile(
                    'https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors',
                    'stable-diffusion-vae-ft-ema-560000-ema-pruned.safetensors'),
                'ckpt': VaeFile(
                    'https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt',
                    'stable-diffusion-vae-ft-ema-560000-ema-pruned.ckpt'),
            }
        },

        'Waifu Diffusion': {
            'v1.4': {
                'kl-f8-anime2 (pruned)': {
                    'ckpt': VaeFile('https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt'),
                },
                'kl-f8-anime (finetuned)': {
                    'ckpt': VaeFile('https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime.ckpt'),
                },
            },
        },

        'TrinArt': {
            'autoencoder_fix_kl-f8-trinart_characters': {
                'ckpt': ModelFile('https://huggingface.co/naclbit/trinart_derrida_characters_v2_stable_diffusion/resolve/main/autoencoder_fix_kl-f8-trinart_characters.ckpt')
            }
        },

        'Anything': {
            # 전부 유출된 노벨AI 의 VAE 와 동일한 파일임
            'v4.0 (same as animevae.pt; unofficial merge)': {
                'animevae.pt': VaeFile('https://huggingface.co/gozogo123/anime-vae/resolve/main/animevae.pt'),
            },
            'v3.0 (same as animevae.pt)': {
                'animevae.pt': VaeFile('https://huggingface.co/gozogo123/anime-vae/resolve/main/animevae.pt'),
            },
        },

        'NovelAI': {
            'animevae.pt': VaeFile('https://huggingface.co/gozogo123/anime-vae/resolve/main/animevae.pt')
        }
    },

    'Textual Inversion (embeddings)': {
        '$sort': True,

        'bad_prompt (negative embedding)': {
            'Version 2': EmbeddingFile('https://huggingface.co/datasets/Nerfgun3/bad_prompt/resolve/main/bad_prompt_version2.pt'),
            'Version 1': EmbeddingFile('https://huggingface.co/datasets/Nerfgun3/bad_prompt/resolve/main/bad_prompt.pt'),
        },
    }
}


def global_disable(disabled: bool):
    for dropdown in dropdowns.children:
        dropdown.disabled = disabled

    download_button.disabled = disabled

    # 마지막 드롭다운이 하위 드롭다운이라면 버튼 비활성화하기
    if not disabled:
        dropdown = dropdowns.children[len(dropdowns.children) - 1]
        download_button.disabled = isinstance(dropdown, dict)


def on_download(_):
    dropdown = dropdowns.children[len(dropdowns.children) - 1]
    entry = dropdown.entries[dropdown.value]

    global_disable(True)

    # 단일 파일 받기
    if isinstance(entry, File):
        entry.download()

    # 다중 파일 받기
    elif isinstance(entry, list):
        for file in entry:
            file.download()

    # TODO: 오류 처리
    else:
        pass

    if DISCONNECT_RUNTIME:
        runtime.unassign()

    global_disable(False)


def on_dropdown_change(event):
    dropdown: widgets.Dropdown = event['owner']
    entries: Union[List, Dict] = dropdown.entries[event['new']]

    # 이전 하위 드롭다운 전부 제거하기
    dropdowns.children = dropdowns.children[:dropdown.children_index + 1]

    if isinstance(entries, dict):
        download_button.disabled = True
        create_dropdown(entries)
        return

    # 하위 드롭다운 만들기
    download_button.disabled = False


def create_dropdown(entries: Dict) -> widgets.Dropdown:
    if '$sort' in entries and entries['$sort'] == True:
        entries = {k: entries[k] for k in sorted(entries)}
        del entries['$sort']

    options = list(entries.keys())
    value = options[0]

    dropdown = widgets.Dropdown(
        options=options,
        value=value)

    setattr(dropdown, 'children_index', len(dropdowns.children))
    setattr(dropdown, 'entries', entries)

    dropdowns.children = tuple(list(dropdowns.children) + [dropdown])

    dropdown.observe(on_dropdown_change, names='value')

    on_dropdown_change({
        'owner': dropdown,
        'new': value
    })

    return dropdown


# 첫 엔트리 드롭다운 만들기
create_dropdown(files)

download_button.on_click(on_download)
