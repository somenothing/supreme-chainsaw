import math
import numpy


class Table:
    def __init__(self, border):
        self.border = border
        self.cue_ball_pos = (0, 0)
        self.target_ball_pos = (0, 0)
        self.all_ball_pos = {}
