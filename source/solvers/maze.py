import typing
from base64 import b85decode
from collections import deque

import cv2
import numpy as np
from detectors.maze import detect_maze
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from game_state import GameState

_B64_MAZES = (
    b'I;j(^Ws_{F(=1ogKB>VLR#E',
    b'vY{JVT9&M((_UK0jI70EWm5',
    b'I*FQuY{lBF$;`~mY{kqrQ&s',
    b'nlh8jI@4^W!7OvrJ~PQSAw>',
    b'GBcYxvqhd#nrxHAtaHI+GgA',
    b's41Jwti@i$T9(?3TEx~iQ&I',
    b'I+L2LVZ~M&TADtEOg70jGgk',
    b'sFRw0Va2R7$!sZCj5ZT?GZO',
    b's56_)8q9vB#f)0g%&bXPRu%')

_MARKERS = (
    (6, 17), (10, 19), (21, 23),
    (0, 18), (16, 33), (4, 26),
    (1, 31), (3, 20), (8, 24))

def _decode_maze(x):
    return np.frombuffer(b85decode(x), dtype=np.uint8)

_MAZES = {
    k: np.stack((v & 0xF, v >> 4), axis=-1).reshape(-1)
    for k, v in zip(_MARKERS, map(_decode_maze, _B64_MAZES))}

_DIRECTIONS = { -6: 'U', -1: 'L', 1: 'R', 6: 'D'}
_UI_BUTTONS = {
    'L': (850, 535), 'R': (1080, 535),
    'U': (970, 425), 'D': (970, 650)}

class Maze:
    def __init__(self, bgr_image: NDArray) -> None:
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        self._start, self._finish, self._marker = detect_maze(hsv_image)
    
    def solve(self, state: 'GameState') -> None:
        path = self._find_path()
        for direction in path:
            state.mov(*_UI_BUTTONS[direction]).ldn().lup().slp()

    def _find_path(self) -> str:
        pred = self._bfs_maze()
        crawl = self._finish
        path = deque([crawl])

        while pred[crawl] != -1:
            path.appendleft(pred[crawl])
            crawl = pred[crawl]
        
        prev = path.popleft()
        result = ''
        while path:
            curr = path.popleft()
            result += _DIRECTIONS[curr - prev]
            prev = curr
        return result

    def _bfs_maze(self) -> list[int]:
        nodes = _MAZES[self._marker]
        queue = deque[int]()
        vis, dist, pred = [False]*36, [100]*36, [-1]*36
        
        vis[self._start] = True
        dist[self._start] = 0
        queue.append(self._start)

        while queue:
            u = queue.popleft()
            for i in range(4):
                if nodes[u] & (1 << i) == 0:
                    continue
                v = u + (-1, 1, -6, 6)[i]
                if vis[v]:
                    continue

                queue.append(v)
                vis[v] = True
                dist[v] = dist[u] + 1
                pred[v] = u
                if v == self._finish:
                    return pred
