import os
import shlex
import time

from pathlib import Path
from urllib.parse import urlparse, unquote
from tempfile import TemporaryDirectory
from typing import Union, List, Dict
from IPython.display import display
from ipywidgets import widgets
from google.colab import drive, runtime

# fmt: off
#@title

#@markdown###<font color="orange">***工作目录路径***</font>
#@markdown模型文件等将被永久保存的目录路径
WORKSPACE = 'SD' #@param {type:"string"}

#@markdown######<font color="orange">***下载完成后,是否会自动结束CORAP运行时?***</font>
DISCONNECT_RUNTIME = True  #@param {type:"boolean"}

# fmt: on

#界面元素
dropdowns = widgets.VBox()
output = widgets.Output()
download_button = widgets.Button(
    description='下载',
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


#文件路径
workspace_dir = Path('drive', 'MyDrive', WORKSPACE)
sd_model_dir = workspace_dir.joinpath('models', 'Stable-diffusion')
sd_embedding_dir = workspace_dir.joinpath('embeddings')
vae_dir = workspace_dir.joinpath('models', 'VAE')

#安装谷歌硬盘
with output:
    drive.mount('drive')

sd_model_dir.mkdir(0o777, True, True)
sd_embedding_dir.mkdir(0o777, True, True)
vae_dir.mkdir(0o777, True, True)


class File:
    prefix: Path

    def __init__(self, url: str, path: os.PathLike = None, *extra_args: List[str]) -> None:
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

        with TemporaryDirectory() as tempdir:
            args = shlex.join((
                '--continue',
                '--always-resume',
                '--summary-interval', '3',
                '--console-log-level', 'error',
                '--max-concurrent-downloads', '16',
                '--max-connection-per-server', '16',
                '--split', '16',
                '--dir', tempdir,
                *self.extra_args,
                self.url
            ))

            with output:
                # 使用aria2获取文件
                # fmt: off
                !which aria2c || apt install -y aria2
                output.clear_output()

                print('使用aria 2接收文件.')
                !aria2c {args}
                output.clear_output()

                print('文件已成功接收,请转到驱动器.')
                print('根据文件大小,此操作可能需要5分钟或更长时间,请稍候.')
                if DISCONNECT_RUNTIME:
                    print('任务完成后将自动关闭运行时,您可以继续执行其他任务.')

                #如果目的地路径不是目录,请按原样使用
                filename = str(self.path) if not self.path.is_dir() else self.path.joinpath(
                    #否则,从文件远程地址获取文件名
                    unquote(os.path.basename(urlparse(self.url).path))
                )

                print(f'路径:{filename}')

                !rsync -aP "{tempdir}/$(ls -AU {tempdir} | head -1)" "{filename}"

                # fmt: on


class ModelFile(File):
    prefix = sd_model_dir


class EmbeddingFile(File):
    prefix = sd_embedding_dir


class VaeFile(File):
    prefix = vae_dir


#型号列表
CONFIG_V2_V = 'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml'

files = {
    'Stable-Diffusion Checkpoints': {
        # 排列并显示当前列表中的键值
        '$sort': True,

        'Stable Diffusion': {
            'v2.1': {
                '768-v': {
                    'ema-pruned': {
                        'safetensors': [
                            ModelFile(
                                'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors',
                                'stable-diffusion-v2-1-786-v-ema-pruned.safetensors'),
                            ModelFile(
                                CONFIG_V2_V, 'stable-diffusion-v2-1-786-v-ema-pruned.yaml'),
                        ],
                        'ckpt': [
                            ModelFile(
                                'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt',
                                'stable-diffusion-v2-1-786-v-ema-pruned.ckpt'),
                            ModelFile(
                                CONFIG_V2_V, 'stable-diffusion-v2-1-786-v-ema-pruned.yaml'),
                        ]
                    },
                    'nonema-pruned': {
                        'safetensors': [
                            ModelFile(
                                'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.safetensors',
                                'stable-diffusion-v2-1-786-v-nonema-pruned.safetensors'),
                            ModelFile(
                                CONFIG_V2_V, 'stable-diffusion-v2-1-786-v-ema-pruned.yaml'),
                        ],
                        'ckpt': [
                            ModelFile(
                                'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.ckpt',
                                'stable-diffusion-v2-1-786-v-nonema-pruned.ckpt'),
                            ModelFile(
                                CONFIG_V2_V, 'stable-diffusion-v2-1-786-v-ema-pruned.yaml'),
                        ],
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
                    'safetensors': [
                        ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.safetensors',
                            'stable-diffusion-v2-0-786-v-ema.safetensors'),
                        ModelFile(
                            CONFIG_V2_V, 'stable-diffusion-v2-1-786-v-ema-pruned.yaml'),
                    ],
                    'ckpt': [
                        ModelFile(
                            'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt',
                            'stable-diffusion-v2-0-786-v-ema.ckpt'),
                        ModelFile(
                            CONFIG_V2_V, 'stable-diffusion-v2-1-786-v-ema-pruned.yaml'),
                    ],
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
                'anime': {
                    'e2': {
                        'fp16': {
                            'safetensors': [
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp16.safetensors'),
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp16.yaml')
                            ],
                            'ckpt': [
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp16.ckpt'),
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp16.yaml')
                            ]
                        },
                        'fp32': {
                            'safetensors': [
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp32.safetensors'),
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp32.yaml')
                            ],
                            'ckpt': [
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp32.ckpt'),
                                ModelFile(
                                    'https://huggingface.co/saltacc/wd-1-4-anime/resolve/main/wd-1-4-epoch2-fp32.yaml')
                            ]
                        },
                    },
                    'e1': {
                        'ckpt': [
                            ModelFile(
                                'https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-1-4-anime_e1.ckpt'),
                            ModelFile(
                                'https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-1-4-anime_e1.yaml'),
                        ]
                    },
                },
                'booru-step-14000-unofficial': {
                    'safetensors': ModelFile('https://huggingface.co/waifu-diffusion/unofficial-releases/resolve/main/wd14-booru-step-14000-unofficial.safetensors'),
                },
            },
            'v1.3.5': {
                '80000-fp32': {
                    'ckpt': ModelFile('https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/wd-1-3-5_80000-fp32.ckpt'),
                },
                'penultimate-ucg-cont': {
                    'ckpt': ModelFile('https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/wd-1-3-penultimate-ucg-cont.ckpt'),
                }
            },
            'v1.3': {
                'fp16': {
                    'ckpt': ModelFile('https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt')
                },
                'fp32': {
                    'ckpt': ModelFile('https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float32.ckpt')
                },
                'full': {
                    'ckpt': ModelFile('https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-full.ckpt')
                },
                'full-opt': {
                    'ckpt': ModelFile('https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-full-opt.ckpt')
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

        'AniReal': {
            'v1.0': {
                'safetensors': ModelFile('https://huggingface.co/Hosioka/AniReal/resolve/main/AniReal.safetensors')
            }
        },

        'OrangeMixs': {
            'AbyssOrangeMix': {
                '2': {
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
            }
        },

        'Protogen': {
            'v8.6 Infinity': {
                'ckpt': ModelFile(
                    'https://huggingface.co/darkstorm2150/Protogen_Infinity_Official_Release/resolve/main/model.ckpt',
                    'ProtoGen_Infinity.ckpt')
            },
            'v8.0 Nova (Experimental)': {
                'ckpt': ModelFile(
                    'https://huggingface.co/darkstorm2150/Protogen_Nova_Official_Release/resolve/main/model.ckpt',
                    'ProtoGen_Nova.ckpt')
            },
            'v7.4 Eclipse (Advanced)': {
                'ckpt': ModelFile(
                    'https://huggingface.co/darkstorm2150/Protogen_Eclipse_Official_Release/resolve/main/model.ckpt',
                    'ProtoGen_Eclipse.ckpt')
            },
            'v5.9 Dragon (RPG themes)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_Dragon_Official_Release/resolve/main/ProtoGen_Dragon.ckpt'),
            },
            'v5.8 (Sci-Fi/Anime)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.8_Official_Release/resolve/main/ProtoGen_X5.8.ckpt'),
            },
            'v5.3 (Photorealism)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x5.3_Official_Release/resolve/main/ProtoGen_X5.3.ckpt'),
            },
            'v3.4 (Photorealism)': {
                'pruned': {
                    'fp16': {
                        'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4-pruned-fp16.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4-pruned-fp16.ckpt'),
                    }
                },
                'safetensors': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4.safetensors'),
                'ckpt': ModelFile('https://huggingface.co/darkstorm2150/Protogen_x3.4_Official_Release/resolve/main/ProtoGen_X3.4.ckpt'),
            },
            'v2.2 (Anime)': {
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
                'v3.0': {
                    'A': {
                        'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v3/7th_anime_v3_A.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v3/7th_anime_v3_A.ckpt'),
                    },
                    'B': {
                        'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v3/7th_anime_v3_B.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v3/7th_anime_v3_B.ckpt'),
                    },
                    'C': {
                        'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v3/7th_anime_v3_C.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v3/7th_anime_v3_C.ckpt'),
                    },
                },
                'v2.0': {
                    'A': {
                        'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_A.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_A.ckpt'),
                    },
                    'B': {
                        'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_B.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_B.ckpt'),
                    },
                    'C': {
                        'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_C.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_C.ckpt'),
                    },
                    'G': {
                        'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_G.safetensors'),
                        'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v2/7th_anime_v2_G.ckpt'),
                    },
                },
                'v1.1': {
                    'safetensors': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v1/7th_anime_v1.1.safetensors'),
                    'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_anime_v1/7th_anime_v1.1.ckpt'),
                },
            },
            'abyss_7th_layer': {
                'G1': {
                    'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_layer/abyss_7th_layerG1.ckpt'),
                },
                'ckpt': ModelFile('https://huggingface.co/syaimu/7th_Layer/resolve/main/7th_layer/Abyss_7th_layer.ckpt')
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
                'kl-f8-anime': {
                    'e2': {
                        'ckpt': VaeFile('https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt'),
                    },
                    'e1': {
                        'ckpt': VaeFile('https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime.ckpt'),
                    }
                },
            },
        },

        'TrinArt': {
            'autoencoder_fix_kl-f8-trinart_characters': {
                'ckpt': ModelFile('https://huggingface.co/naclbit/trinart_derrida_characters_v2_stable_diffusion/resolve/main/autoencoder_fix_kl-f8-trinart_characters.ckpt')
            }
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

    # 如果最后一个下拉菜单是子下拉菜单,请禁用按钮
    if not disabled:
        dropdown = dropdowns.children[len(dropdowns.children) - 1]
        download_button.disabled = isinstance(dropdown, dict)


def on_download(_):
    dropdown = dropdowns.children[len(dropdowns.children) - 1]
    entry = dropdown.entries[dropdown.value]

    global_disable(True)

    #接收单个文件
    if isinstance(entry, File):
        entry.download()

    #接收多个文件
    elif isinstance(entry, list):
        for file in entry:
            file.download()

    #TODO:错误处理
    else:
        pass

    if DISCONNECT_RUNTIME:
        print('文件已成功移动,现在可以关闭运行时了.')

        #如果立即退出运行时,最后一次输出将被截断
        time.sleep(1)
        runtime.unassign()

    global_disable(False)


def on_dropdown_change(event):
    dropdown: widgets.Dropdown = event['owner']
    entries: Union[List, Dict] = dropdown.entries[event['new']]

    #删除所有上一个子下拉列表
    dropdowns.children = dropdowns.children[:dropdown.children_index + 1]

    if isinstance(entries, dict):
        download_button.disabled = True
        create_dropdown(entries)
        return

    #创建子下拉列表
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


#创建第一个条目下拉列表
create_dropdown(files)

download_button.on_click(on_download)
