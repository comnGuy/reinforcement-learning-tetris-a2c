# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:35:26 2018

@author: Bernhard Preisler
"""

#!/usr/bin/env python2
#-*- coding: utf-8 -*-

# NOTE FOR WINDOWS USERS:
# You can download a "exefied" version of this game at:
# http://hi-im.laria.me/progs/tetris_py_exefied.zip
# If a DLL is missing or something like this, write an E-Mail (me@laria.me)
# or leave a comment on this gist.

# Very simple tetris implementation
# 
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop
#
# Have fun!

# Copyright (c) 2010 "Laria Carolin Chabowski"<me@laria.me>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from random import randrange as rand
import pygame, sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from collections import deque, Counter

import copy
import random

show_results = False
load_model = False

# The configuration
cell_size =    18
#cols =        10
#rows =        22
#cols =        6
#rows =        10
maxfps = 500

colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35) # Helper color for background grid
]

# Define the shapes of the single parts
"""tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],
    
    [[0, 2, 2],
     [2, 2, 0]],
    
    [[3, 3, 0],
     [0, 3, 3]],
    
    [[4, 0, 0],
     [4, 4, 4]],
    
    [[0, 0, 5],
     [5, 5, 5]],
    
    [[6, 6, 6, 6]],
    
    [[7, 7],
     [7, 7]]
]"""

tetris_shapes = [    
    [[7, 7],
     [7, 7]], 
    [[7, 7],
     [7, 7]], 
    [[7, 7],
     [7, 7]], 
    [[7, 7],
     [7, 7]], 
    [[7, 7],
     [7, 7]], 
    [[7, 7],
     [7, 7]], 
    [[7, 7],
     [7, 7]]
]

def rotate_clockwise(shape):
    return [ [ shape[y][x]
            for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1) ]

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[ cy + off_y ][ cx + off_x ]:
                    return True
            except IndexError:
                return True
    return False

#def remove_row(board, row):
#    del board[row]
#    return [[0 for i in range(cols)]] + board
    
def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1    ][cx+off_x] += val
    return mat1

#def new_board():
#    board = [ [ 0 for x in range(cols) ]
#            for y in range(rows) ]
#    board += [[ 1 for x in range(cols)]]
#    return board


class Tetris:
    
    def __init__(self, field = (11, 6)):
        self.field = field
        self.gameover = False
        self.paused = False
        self.past_reward = 0
        self.cleard_lines = 0
        self.width = cell_size*(self.field[1]+6)
        self.height = cell_size*self.field[0]
        self.rlim = cell_size*self.field[1]
        self.bground_grid = [[ 8 if x%2==y%2 else 0 for x in range(self.field[1])] for y in range(self.field[0])]
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.init_game()
    
    def new_stone(self):
        self.stone = self.next_stone[:]
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.stone_x = int(self.field[1] / 2 - len(self.stone[0])/2)
        self.stone_y = 0
        
        if check_collision(self.board,
                           self.stone,
                           (self.stone_x, self.stone_y)):
            self.gameover = True
            
    def new_board(self):
        board = [ [ 0 for x in range(self.field[1]) ]
                for y in range(self.field[0]-1) ]
        board += [[ 1 for x in range(self.field[1])]]
        return board
    
    def init_game(self):
        self.board = self.new_board()
        self.new_stone()
        self.level = 1
        self.score = 0
        self.lines = 0
        self.sum_lines = 0
        self.gameover = False
        #pygame.time.set_timer(pygame.USEREVENT+1, 1000)
        
    def remove_row(self, board, row):
        del board[row]
        return [[0 for i in range(self.field[1])]] + board
    
    def disp_msg(self, msg, topleft):
        x,y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255,255,255),
                    (0,0,0)),
                (x,y))
            y+=14
    
    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image =  self.default_font.render(line, False,
                (255,255,255), (0,0,0))
        
            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2
        
            self.screen.blit(msg_image, (
              self.width // 2-msgim_center_x,
              self.height // 2-msgim_center_y+i*22))
    
    def draw_matrix(self, matrix, offset):
        off_x, off_y  = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x+x) *
                              cell_size,
                            (off_y+y) *
                              cell_size, 
                            cell_size,
                            cell_size),0)
    
    def add_cl_lines(self, n):
        self.cleard_lines = n
        self.sum_lines += n
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level*6:
            self.level += 1
            newdelay = 1000-50*(self.level-1)
            newdelay = 100 if newdelay < 100 else newdelay
            #pygame.time.set_timer(pygame.USEREVENT+1, newdelay)

    # -1 oder 1
    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > self.field[1] - len(self.stone[0]):
                new_x = self.field[1] - len(self.stone[0])
            if not check_collision(self.board,
                                   self.stone,
                                   (new_x, self.stone_y)):
                self.stone_x = new_x
            else:
                self.collion_side = True
    def quit(self):
        self.center_msg("Exiting...")
    
    def drop(self, manual):
        if not self.gameover and not self.paused:
            self.score += 1 if manual else 0
            self.stone_y += 1
            if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                  self.board,
                  self.stone,
                  (self.stone_x, self.stone_y))
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = self.remove_row(
                              self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False
    
    def insta_drop(self):
        if not self.gameover and not self.paused:
            while(not self.drop(True)):
                pass
    
    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone
    
    def toggle_pause(self):
        self.paused = not self.paused
    
    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False
            
    def get_action_via_number(self, key_number):
        if key_number == 0:
            self.move(-1)
        if key_number == 1:
            self.move(1)
        if key_number == 2:
            self.drop(True)
        #if key_number == 3:
        #    return 'UP'
        #if key_number == 3:
        #    return 'RETURN'
        return 'NOTHING'
    
    def get_board_dim(self):
        return (1, len(self.board), len(self.board[0]))

    def merge_board_stone(self, board_i, stone_i, x_i, y_i):
        count_x = 0
        count_y = 0
        for slice_stone in stone_i:
            for point in slice_stone:
                board_i[y_i+count_y][x_i+count_x] = point
                count_x += 1
            count_y += 1
            count_x = 0
        #print(board_i)
        board_i = np.asarray(board_i)
        #print(board_i.shape[0])
        return np.reshape(board_i, [1, 1, board_i.shape[0], board_i.shape[1]])
    
    def unnest_state(self, state):
        tmp_list = []
        for state_flat in state[0][0]:
            for state_flat_2 in state_flat:
                tmp_list.append(state_flat_2)
        return tmp_list

    def reward(self):
        #reward = self.score - self.past_reward
        #self.past_reward = self.score
        reward = self.cleard_lines * 3
        self.cleard_lines = 0
        # Game ends?
        if self.score > 5000:
            self.gameover = True
        reward = reward if not self.gameover else -1
        #print(reward)
        return reward
        
    def print_state(self, state):
        print('Start state')
        for line in state[0]:
            #for cell in 
            #print(line)
            for cell in line:
                print(cell, end=" ")
            print('')
        print('End state')
    
    def nothing(self):
        pass
    
    def get_env(self):
        board = copy.deepcopy(self.board)
        return self.merge_board_stone(board, self.stone, self.stone_x, self.stone_y)
    
    def act(self, act):
        self.get_action_via_number(act)
        
    def status(self):
        return self.gameover
