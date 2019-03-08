import os
import json

from typing import Iterable, List, Dict

from corvid.table.table import Cell, Table
from corvid.table.table_loader import CellLoader, TableLoader
from corvid.util.strings import format_grid
from corvid.util.geom import Box


class OmnipageTableLoaderException(Exception):
    pass


class OmnipageCell(Cell):
    def __init__(self,
                 tokens: List[str],
                 index_topleft_row: int,
                 index_topleft_col: int,
                 rowspan: int,
                 colspan: int,
                 bounding_box: Box):
        self.bounding_box = bounding_box
        super(OmnipageCell, self).__init__(tokens=tokens,
                                           index_topleft_row=index_topleft_row,
                                           index_topleft_col=index_topleft_col,
                                           rowspan=rowspan,
                                           colspan=colspan)

    def to_json(self) -> Dict:
        """Serialize to JSON dictionary"""
        json = super(OmnipageCell, self).to_json()
        json.update({'bounding_box': self.bounding_box.to_json()})
        return json


class OmnipageTable(Table):
    def __init__(self,
                 paper_id: str,
                 index_page: int,
                 index_on_page: int,
                 caption: str,
                 bounding_box: Box,
                 grid: Iterable[Iterable[OmnipageCell]] = None,
                 cells: Iterable[OmnipageCell] = None,
                 nrow: int = None,
                 ncol: int = None):
        """For a given `paper_id`, a table is uniquely identified by its
        `index_page` and `index_on_page`
        """
        self.paper_id = paper_id
        self.index_page = index_page
        self.index_on_page = index_on_page
        self.id = '_'.join([str(self.index_page), str(self.index_on_page)])
        self.caption = caption
        self.bounding_box = bounding_box
        super(OmnipageTable, self).__init__(grid=grid,
                                            cells=cells, nrow=nrow, ncol=ncol)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = ''
        s += 'PAPER: {}\n'.format(self.paper_id)
        s += 'PAGE: {}\n'.format(self.index_page + 1)
        s += 'INDEX: {}\n\n'.format(self.index_on_page)
        s += format_grid([[str(cell) for cell in row]
                          for row in self.grid]) + '\n\n' + self.caption
        return s

    def to_json(self) -> Dict:
        """Serialize to JSON dictionary"""
        json = super(OmnipageTable, self).to_json()
        json.update({
            'paper_id': self.paper_id,
            'index_page': self.index_page,
            'index_on_page': self.index_on_page,
            'caption': self.caption,
            'bounding_box': self.bounding_box.to_json()
        })
        return json


class OmnipageCellLoader(CellLoader):
    def __init__(self):
        super(OmnipageCellLoader, self).__init__(cell_type=OmnipageCell)

    def from_json(self, json: Dict) -> OmnipageCell:
        d = json.copy()
        d['bounding_box'] = Box.from_json(d['bounding_box'])
        return super(OmnipageCellLoader, self).from_json(json=d)


class OmnipageTableLoader(TableLoader):
    def __init__(self):
        self.cell_loader = OmnipageCellLoader()
        super(OmnipageTableLoader, self).__init__(
            table_type=OmnipageTable,
            cell_loader=self.cell_loader
        )

    def from_json(self, json: Dict) -> OmnipageTable:
        try:
            d = json.copy()
            d['bounding_box'] = Box.from_json(d['bounding_box'])
            return super(OmnipageTableLoader, self).from_json(json=d)
        except Exception as e:
            print(e)
            raise OmnipageTableLoaderException


class PaperTable(object):
    def __init__(self, id: str, papers_dir: str):
        self.id = id
        self.papers_dir = papers_dir
        self._tables = None

    @property
    def omnipage_xml_to_tables_json_path(self) -> str:
        path = '{}.json'.format(os.path.join(self.papers_dir, self.id))
        return path

    @property
    def tables(self) -> List[OmnipageTable]:
        if self._tables is None:
            try:
                table_loader = OmnipageTableLoader()
                with open(self.omnipage_xml_to_tables_json_path, 'r') as f:
                    self._tables = [table_loader.from_json(json=d)
                                    for d in json.load(f)]
            except FileNotFoundError as e:
                pass
            except OmnipageTableLoaderException as e:
                pass

        return self._tables

    @property
    def s2_url(self) -> str:
        return 'https://semanticscholar.org/paper/{}'.format(self.id)