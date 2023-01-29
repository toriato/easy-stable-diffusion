import itertools

from typing import Any, Iterable, List, Dict, Optional, Callable
from pathlib import Path
from tempfile import mkdtemp
from ipywidgets import widgets

from .utils import wrap_widget_locks


class FileSelectorOption:
    def __init__(
        self,
        name: str,
        extractor: Optional[Callable[['FileSelectorOption'], Any]] = None,
    ) -> None:
        self.name = name
        self.extractor = extractor

    def create(self, selector: 'FileSelector') -> Optional[widgets.Widget]:
        pass

    def selected(self, selector: 'FileSelector'):
        pass

    def deselected(self, selector: 'FileSelector'):
        pass

    def extract(self) -> Any:
        if self.extractor:
            return self.extractor(self)
        return self.name


class FileSelectorWidget(FileSelectorOption):
    def __init__(
        self,
        name: str,
        widget: Optional[widgets.Widget] = None,
        extractor: Optional[Callable[['FileSelectorOption'], Any]] = None
    ) -> None:
        super().__init__(name, extractor)
        self.widget = widget

    def selected(self, _):
        if self.widget:
            self.widget.layout.display = 'inherit'  # type: ignore

    def deselected(self, _):
        if self.widget:
            self.widget.layout.display = 'none'  # type: ignore


class FileSelectorDownloader(FileSelectorWidget):
    def __init__(
        self,
        name='< 인터넷 주소로부터 파일 다운로드 >',
        default_url='https://...',
        extractor: Optional[Callable[['FileSelectorOption'], Any]] = None,
    ) -> None:
        super().__init__(
            name,
            widgets.Text(
                value=default_url,
                layout={'width': 'auto'}
            ),
            extractor
        )

    def extract(self):
        raise NotImplementedError()


class FileSelector:
    """
    로컬 파일 시스템 또는 인터넷으로부터 파일을 선택하거나 업로드할 수 있는 위젯 집합을 만듭니다
    """
    dropdown: widgets.Dropdown
    refresh_button: widgets.Button

    or_upload: Optional[widgets.FileUpload] = None
    or_text: Optional[widgets.Text] = None

    lock_group: List[object] = []

    def __init__(
        self,
        options: Iterable[FileSelectorOption] = [],
        path_root: Optional[Path] = None,
        path_globs: Iterable[str] = (),
        path_generator: Optional[Callable[..., List[Path]]] = None,
    ) -> None:
        """
        :param options: 드롭다운 추가할 옵션들
        :param path_root: 파일 검색을 시작할 디렉터리 경로
        :param path_globs: 파일 검색에 사용할 glob 패턴
        :param path_generator: 파일 검색에 사용할 함수
        """
        self.dropdown = widgets.Dropdown(
            options=[(option.name, option) for option in options],
            # margin 이나 padding 등의 속성 때문에 전체 폭보다 조금 벗어나므로 조금 빼줌
            # border-box 를 사용하면 해결할 수 있으나 귀찮음...
            layout={'width': 'calc(100% - 5px)'})

        self.refresh_button = widgets.Button(
            description='🔄',
            layout={'width': 'auto'})

        self.lock_group += [self.dropdown, self.refresh_button]

        def on_update_dropdown(change: Dict[str, Any]) -> None:
            """
            옵션 값이 변경될 때 각 옵션 객체에게 이벤트를 던져주는 함수
            """
            old = change['old']
            new = change['new']

            if old and isinstance(old, FileSelectorOption):
                old.deselected(self)

            if new and isinstance(new, FileSelectorOption):
                new.selected(self)

        def on_click_refresh_button(_) -> None:
            paths: List[Path]

            # 사용자가 제공한 함수로부터 경로 가져오기
            if path_generator:
                paths = path_generator()

            # 로컬 파일 시스템에서 하위 경로 가져오기
            else:
                assert path_root, '루트 디렉터리 경로가 선언되지 않았습니다, 하위 경로를 가져오려면 루트 경로가 필요합니다'
                glob = path_root.glob

                # glob 패턴을 통해 일치하는 모든 하위 파일 목록 가져오기
                path_chunks = map(
                    lambda pattern: [p for p in glob(pattern)],
                    path_globs
                )

                # 2차원 배열을 1차원 배열로 펼치기
                paths = list(itertools.chain(*path_chunks))

            # 기본 옵션 + 새로 찾은 경로 목록 추가하기
            self.dropdown.options = tuple([
                (opt.name, opt)
                for opt in [FileSelectorOption(str(p)) for p in paths] + list(options)
            ])

        # 각 위젯에 이벤트 핸들러 연결
        self.dropdown.observe(
            wrap_widget_locks(
                on_update_dropdown,
                self.lock_group
            ),
            names='value'  # type: ignore
        )

        self.refresh_button.on_click(
            wrap_widget_locks(
                on_click_refresh_button,
                self.lock_group
            ))

    def create_ui(self) -> widgets.Box:
        """
        위젯 집합을 담고 있는 박스 위젯을 만듭니다
        """
        option_widgets = [
            opt.widget
            for _, opt in self.dropdown.options  # type: ignore
            if isinstance(opt, FileSelectorWidget)
        ]

        return widgets.VBox((
            widgets.GridBox(
                (widgets.Box((self.dropdown,)), self.refresh_button),
                layout={'grid_template_columns': '5fr 1fr'}
            ),
            *option_widgets
        ))

    def save_uploaded_file(self) -> Path:
        """
        사용자가 업로드한 파일을 메모리에서 로컬 파일 시스템으로 옮긴 뒤 경로를 반환합니다

        :return: 임시 파일 경로
        """
        assert self.or_upload, '업로드가 허용되지 않은 파일 선택자입니다'

        # TODO: 작업 종료 후 임시 폴더 제거하기
        temp_dir = Path(mkdtemp())

        assert isinstance(self.or_upload.value, dict), '업로드된 파일이 없습니다'
        assert len(self.or_upload.value) < 2, '파일이 하나 이상 업로드 됐습니다'

        data = self.or_upload.value[0]
        path = temp_dir.joinpath(data.name)

        with path.open() as file:
            # TODO: 용량 큰 파일 대응
            file.write(data.content)

        return path

    def download_file(self) -> Path:
        """
        사용자가 입력한 주소로부터 파일을 다운로드한 뒤 로컬 파일 시스템에 저장하고 경로를 반환합니다

        :return: 임시 파일 경로
        """
        raise NotImplementedError()

    def extract(self) -> Optional[str]:
        """
        사용자가 선택한 파일의 경로를 가져옵니다

        Returns: 사용자가 선택한 파일의 문자열 경로
        """
        assert isinstance(self.dropdown.value, FileSelectorOption), ''

        value = self.dropdown.value.extract()

        if isinstance(value, Path):
            value = str(value)

        if value is None:
            return value

        assert isinstance(value, str), f'옵션의 자료형이 문자열이 아니고 {type(value)} 입니다'
        return value
