"""

The Table class is a physical representation of tables extracted from documents

This code was written by Kyle Lo.

https://github.com/allenai/corvid/edit/master/corvid/table/
"""

from typing import List, Dict, Tuple, Union, Iterable, Callable

import numpy as np


def format_grid(grid: List[List[str]]) -> str:
    """
    e.g.
    input: [['a', 'b', 'c'], ['d', 'e', 'f']]
    output: 'a\tb\tc\nd\te\tf'
    printed:
    >    a   b   c
    >    d   e   f
    Source: https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list"""
    if any([len(row) != len(grid[0]) for row in grid]):
        raise Exception('Grid missing entries (i.e. different row lengths)')
    g = [[cell if len(cell) > 0 else ' ' for cell in row] for row in grid]
    lens = [max(map(len, col)) for col in zip(*g)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    return '\n'.join([fmt.format(*row) for row in g])


class Cell(object):
    """A Cell is a single unit of data in a Table separated from other Cells
    by whitespace and/or lines.  A Cell corresponds to its own row and
    column index (or indices) disjoint from those of other Cells."""

    def __init__(self,
                 tokens: List[str],
                 index_topleft_row: int,
                 index_topleft_col: int,
                 rowspan: int = 1,
                 colspan: int = 1):
        self.tokens = tokens
        self.index_topleft_row = index_topleft_row
        self.index_topleft_col = index_topleft_col
        self.rowspan = rowspan
        self.colspan = colspan

    def __repr__(self):
        return ' '.join([str(token) for token in self.tokens])

    def __str__(self):
        return ' '.join([str(token) for token in self.tokens])

    @property
    def indices(self) -> List[Tuple[int, int]]:
        """Returns a list of indices for iterating over the Cell in
        top-to-bottom, left-to-right order.  For example, a 2x2 Cell starting
        at [1, 2] will return [(1, 2), (1, 3), (2, 2), (2, 3)]."""
        return [(self.index_topleft_row + i, self.index_topleft_col + j)
                for i in range(self.rowspan) for j in range(self.colspan)]

    def to_json(self) -> Dict:
        """Serialize to JSON dictionary"""
        json = {
            'tokens': self.tokens,
            'index_topleft_row': self.index_topleft_row,
            'index_topleft_col': self.index_topleft_col,
            'rowspan': self.rowspan,
            'colspan': self.colspan
        }
        return json


# TODO: consider analogous method that returns all indices given a (multispan) cell
class Table(object):
    """A Table is a collection of Cells.  Visually, it may look like:

    >        |          header          |
    >        |  col1  |  col2  |  col3  |
    >  row1  |    a   |    b   |    c   |
    >  row2  |    d   |    e   |    f   |

    These Cells can be stored in two formats within a Table:

    (*) A grid-style format mimics a 2D matrix-like representation.
        It provides the concept of rows and columns, which
        induces a method of indexing using an [i,j] operator.

        Multirow/column Cells are treated in the grid-style format as having
        multiple indices that return the same Cell object.  For example:

    >  [0,0]       |  [0,1] header  |  [0,2] header  |  [0,3] header  |
    >  [1,0]       |  [1,1]  col1   |  [1,2]  col2   |  [1,3]  col3   |
    >  [2,0] row1  |  [2,1]   a     |  [2,2]   b     |  [2,3]   c     |
    >  [3,0] row2  |  [3,1]   d     |  [3,2]   e     |  [3,3]   f     |

       s.t. the same `header` Cell can be access via `[0,1]`, `[0,2]` or `[0,3]`


    (*) A list-style format handles multirow/column Cells by treating
        each Cell object (regardless of its size) as an individual item read
        from the Table in left-to-right, top-to-down fashion.

    >  [0]              > [5]  col3        > [10]  row2
    >  [1]  header      > [6]  row1        > [11]  d
    >  [2]              > [7]  a           > [12]  e
    >  [3]  col1        > [8]  b           > [13]  f
    >  [4]  col2        > [9]  c

       Here, each Cell is treated as a single element of the list, regardless
       of its row/colspan.
    """

    def __init__(self,
                 grid: Iterable[Iterable[Cell]] = None,
                 cells: Iterable[Cell] = None,
                 nrow: int = None, ncol: int = None):
        assert bool(grid is not None) ^ bool(cells and nrow and ncol)
        if grid is not None:
            self.grid = np.array(grid)
            assert self.nrow > 0 and self.ncol > 0
            self.cells = self._cells_from_grid(grid=self.grid)
        if cells is not None:
            self.cells = list(cells)
            self.grid = self._grid_from_cells(cells=self.cells,
                                              nrow=nrow, ncol=ncol)

    @property
    def nrow(self) -> int:
        return self.grid.shape[0]

    @property
    def ncol(self) -> int:
        return self.grid.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def __getitem__(self, index: Union[int, slice, Tuple]) -> \
            Union[Cell, List[Cell]]:
        """Indexes Table elements via its grid:
            * [int, int] returns a single Cell
            * [slice, int] or [int, slice] returns a List[Cell]

            or via its cells:
            * [int] returns a single Cell
            * [slice] returns a List[Cell]
        """
        if isinstance(index, tuple):
            grid = self.grid[index]
            if isinstance(grid, Cell):
                return grid
            elif len(grid.shape) == 1:
                return grid.tolist()
            else:
                raise IndexError('Not supporting [slice, slice]')
        elif isinstance(index, int) or isinstance(index, slice):
            return self.cells[index]
        else:
            raise IndexError('Only integers and slices')

    def __repr__(self):
        return str(self)

    def __str__(self):
        return format_grid([[str(cell) for cell in row] for row in self.grid])

    def _cells_from_grid(self, grid: np.ndarray) -> List[Cell]:
        """Create List[Cell] from a 2D numpy array of Cells"""
        cells = []
        nrow, ncol = grid.shape
        is_visited = [[False for _ in range(ncol)] for _ in range(nrow)]
        for i in range(nrow):
            for j in range(ncol):
                if is_visited[i][j]:
                    continue
                else:
                    cell = grid[i, j]
                    cells.append(cell)
                    for index_row, index_col in cell.indices:
                        is_visited[index_row][index_col] = True
        return cells

    def _grid_from_cells(self, cells: List[Cell],
                         nrow: int, ncol: int) -> np.ndarray:
        """Create 2D numpy array of Cells from a list of Cells & dimensions"""
        grid = np.array([[None for _ in range(ncol)] for _ in range(nrow)])

        for cell in cells:
            for i, j in cell.indices:
                if grid[i, j] is None:
                    grid[i, j] = cell
                else:
                    raise ValueError('Multiple cells inserted into grid[{},{}]'
                                     .format(i, j))

        for i in range(nrow):
            for j in range(ncol):
                if grid[i, j] is None:
                    raise ValueError('No cell in grid[{},{}]'.format(i, j))

        # index_row, index_col = 0, 0
        # for cell in cells:
        #     # insert copies of cell into grid based on its row/colspan
        #     for i in range(index_row, index_row + cell.rowspan):
        #         for j in range(index_col, index_col + cell.colspan):
        #             grid[i, j] = cell
        #
        #     # update `index_row` and `index_col` by scanning for next empty cell
        #     # jump index to next row if reach the right-most column
        #     while index_row < nrow and grid[index_row, index_col]:
        #         index_col += 1
        #         if index_col == ncol:
        #             index_col = 0
        #             index_row += 1
        #
        # # check that grid is complete (i.e. fully populated with cells)
        # if not grid[-1, -1]:
        #     raise ValueError('Cells dont fill out the grid')

        return grid

    def to_json(self) -> Dict:
        """Serialize to JSON dictionary"""
        json = {
            'cells': [c.to_json() for c in self.cells],
            'nrow': self.nrow,
            'ncol': self.ncol
        }
        return json


"""
Handles loading of Tables from inputs other than `grid` or `cells`.
Loader design chosen because easier to work with when start subclassing Cell
and Table as opposed to needing to override @classmethods
"""

class CellLoader(object):
    def __init__(self,
                 cell_type: Callable[..., Cell]):
        self.cell_type = cell_type

    def from_json(self, json: Dict) -> Cell:
        cell = self.cell_type(**json)
        return cell


class TableLoader(object):
    def __init__(self,
                 table_type: Callable[..., Table],
                 cell_loader: CellLoader):
        self.table_type = table_type
        self.cell_loader = cell_loader

    def from_json(self, json: Dict) -> Table:
        cells = [self.cell_loader.from_json(d) for d in json['cells']]
        kwargs = {k: v for k, v in json.items() if k not in 'cells'}
        table = self.table_type(cells=cells, **kwargs)
        return table