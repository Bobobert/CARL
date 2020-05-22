#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:16:03 2020

@author: ebecerra
"""

# Math
import numpy as np
import math
import time

# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import imageio

# Dependencies
import forest_fire


# Features to add

# Action info attribute, DONE
# Rocks, dead cells
# Rock boundary
# Custom starting positions
# Water feature

class Helicopter(forest_fire.ForestFire):
    """ Helicopter class """
    def __init__(self, pos_row = None, pos_col = None, freeze = None, water = 100,
                 n_row = 16, n_col = 16,
                 p_tree=0.100, p_fire=0.001, p_init_tree=0.75,
                 boundary='reflective', tree = 3, empty = 1, fire = 7, 
                 gif_count=0):
        super().__init__(n_row,n_col,
             p_tree,p_fire,p_init_tree,
             boundary,tree,empty,fire)
        # Helicopter attributes
        self.actions_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.actions_cardinality = len(self.actions_set)
        self.checkpoints = []
        self.checkpoint_counter = 0
        self.frames = [] # Potential high memory usage
        self.gif = gif_count
        self.hits_round = 0
        if pos_row is None:
            # Start aprox in the middle
            self.pos_row = math.ceil(self.n_row/2) - 1
        else:
            self.pos_row = pos_row
        if pos_col is None:
            # Start aprox in the middle
            self.pos_col = math.ceil(self.n_col/2) - 1
        else:
            self.pos_col = pos_col      
        if freeze is None:
            # Move aprox a 1/4 of the grid
            self.freeze = math.ceil((self.n_row + self.n_col) / 4)
        else:
            self.freeze = freeze
        self.defrost = self.freeze
        # Water not yet implemented
        self.water = water
        # Number of Times where a fire was extinguished
        self.hits = 0
        # Render Info
        self.current_reward = 0
        self.color_tree = np.array([15, 198, 43, int(1.0*255)]) # Green RGBA
        self.color_empty = np.array([255, 245, 166, int(1.0*255)]) # Beige RGBA
        self.color_fire = np.array([255, 106, 58, int(1.0*255)]) # Red RGBA
        self.grid_to_rgba()
    def step(self, action):
        """Must return tuple with
        numpy array, int reward, bool termination, dict info
        """
        termination = False
        if self.defrost != 0:
            self.new_pos(action)
            self.hits_round = self.extinguish_fire()
            self.hits += self.hits_round
            self.defrost -= 1
            obs = (self.grid, np.array([self.pos_row, self.pos_col]))
            # Don't delay the reward
            # Reward it often
            reward = self.calculate_reward()
            self.current_reward = reward
            return (obs, reward, termination, {'hits': self.hits})
        if self.defrost == 0:
            # Run fire simulation
            self.update()
            self.new_pos(action)
            self.hits_round = self.extinguish_fire()
            self.hits += self.hits_round
            self.defrost = self.freeze
            obs = (self.grid, np.array([self.pos_row, self.pos_col]))
            reward = self.calculate_reward()
            self.current_reward = reward
            return ((obs, reward, termination, {'hits': self.hits}))
    def calculate_reward(self):
        reward = 0
        for row in range(self.n_row):
            for col in range(self.n_col):
                if self.grid[row][col] == self.fire:
                    reward += 9 # Making the reward on the negative sense to calculate the cost to minimize.
        reward -= 2*self.hits_round # Resting a point per fire put out
        self.hits_round = 0 
        return reward
    def new_pos(self, action):
        self.pos_row = self.pos_row if action == 5\
            else self.pos_row if self.is_out_borders(action, pos='row')\
            else self.pos_row - 1 if action in [1,2,3]\
            else self.pos_row + 1 if action in [7,8,9]\
            else self.pos_row
        self.pos_col = self.pos_col if action == 5\
            else self.pos_col if self.is_out_borders(action, pos='col')\
            else self.pos_col - 1 if action in [1,4,7]\
            else self.pos_col + 1 if action in [3,6,9]\
            else self.pos_col
        return (self.pos_row, self.pos_col)
    def is_out_borders(self, action, pos):
        if pos == 'row':
            # Check Up movement
            if action in [1,2,3] and self.pos_row == 0:
                out_of_border = True
            # Check Down movement
            elif action in [7,8,9] and self.pos_row == self.n_row-1:
                out_of_border = True
            else:
                out_of_border = False
        elif pos == 'col':
            # Check Left movement
            if action in [1,4,7] and self.pos_col == 0:
                out_of_border = True
            # Check Right movement
            elif action in [3,6,9] and self.pos_col == self.n_col-1:
                out_of_border = True
            else:
                out_of_border = False
        else:
            raise "Argument Error: pos = str 'row'|'col'"
        return out_of_border
    def extinguish_fire(self):
        """Check where the helicopter is
        then extinguish at that place"""
        hit = 0
        row = self.pos_row
        col = self.pos_col
        current_cell = self.grid[row][col]
        if current_cell == self.fire:
            self.grid[row][col] = self.empty
            hit = 1
        return hit
    def reset(self):
        # Another random grid
        self.__init__(None,None,
                      self.freeze,self.water,
                      self.n_row,self.n_col,
                      self.p_tree,self.p_fire,self.p_init_tree,
                      self.boundary,self.tree,self.empty,self.fire,
                      self.gif)
        # Return first observation
        return (self.grid, np.array([self.pos_row,self.pos_col]))
    def grid_to_rgba(self):
        rgba_mat = self.grid.tolist()
        for row in range(self.n_row):
            for col in range(self.n_col):
                if rgba_mat[row][col] == self.tree:
                    rgba_mat[row][col] = self.color_tree
                elif rgba_mat[row][col] == self.empty:
                    rgba_mat[row][col] = self.color_empty
                elif rgba_mat[row][col] == self.fire:
                    rgba_mat[row][col] = self.color_fire
                else:
                    raise 'Error: unidentified cell'
        rgba_mat = np.array(rgba_mat)
        self.rgba_mat = rgba_mat
        return rgba_mat

    def render_frame(self, show=True, wait_time=-1, title=''):
        # Plot style
        sns.set_style('whitegrid')
        # Main Plot
        if show and wait_time > 0:
            plt.ion()
        fig = plt.imshow(self.grid_to_rgba(), aspect='equal', animated=True)
        # Title showing Reward
        if title ==  '':
            plt.title('Reward {}'.format(self.current_reward))
        else:
            plt.title(title)
        # Modify Axes
        ax = plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, self.n_col, 1))
        ax.set_yticks(np.arange(0, self.n_row, 1))
        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, self.n_col, 1))
        ax.set_yticklabels(np.arange(0, self.n_row, 1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, self.n_col, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.n_row, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=2)
        ax.grid(which='major', color='w', linestyle='-', linewidth=0)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        # Mark the helicopter position
        # Put a red X
        plt.scatter(self.pos_col, self.pos_row,
                    marker='x', color='red',
                    s=50, linewidths=50,
                    zorder=11)
        fig = plt.gcf()
        if show:
            if wait_time > 0:
                plt.draw()
                plt.pause(wait_time)
                plt.close('all')
            else:
                plt.show()
        return fig

    def frame(self, title=''):
        # Saves a frame on the buffer of frames of the object
        fig = self.render_frame(show=False, title=title)
        fig.canvas.draw()      # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.clf()
        self.frames.append(image)
        return None

    def render(self, fps=5, flush=True):
        # Function to generate gif images of a sequence of Frames
        self.gif += 1
        self.frame(title='Fin!')
        imageio.mimsave("./Runs/Helicopter_{0}_{1}.gif".format(self.gif, 
                        time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime())), 
                        self.frames, fps=fps)
        if flush:
            self.frames = []
        return None

    def close(self):
        #print('Gracefully Exiting, come back soon')
        return None

    def __del__(self):
        return None

    def make_checkpoint(self): 
        # Function to save a state of the environment to branch
        self.checkpoints.append((self.grid, self.pos_row, self.pos_col, self.defrost, self.hits, self.current_reward))
        self.checkpoint_counter += 1
        return self.checkpoint_counter - 1
    
    def load_checkpoint(self, checkpoint_id):
        # Function to recall a previous state of the environment given the id
        try:
            self.grid, self.pos_row, self.pos_col, self.defrost, self.hits, self.current_reward = self.checkpoints[checkpoint_id]
        except:
            raise Exception("Checkpoint_id must be invalid.")
    
    def Copy(self):
        # Does a simple copy to use for sample recollection in parallel of the object
        NEW = Helicopter(pos_row = self.pos_row, pos_col = self.pos_col, freeze = self.freeze, water = self.water,
                 n_row = self.n_row, n_col = self.n_col,
                 p_tree=self.p_tree, p_fire=self.p_fire, p_init_tree=0.75,
                 boundary=self.boundary, tree = self.tree, empty = self.empty, fire = self.fire)
        NEW.grid = self.grid
        NEW.checkpoints = self.checkpoints
        NEW.checkpoint_counter = self.checkpoint_counter
        NEW.current_reward = self.current_reward
        return NEW

    def Encode(self):
        # Enconding strings for the grid just in line form
        s = str(self.pos_row) + str(self.pos_col) 
        for j in range(self.n_row):
            for i in range(self.n_col):
                cell = self.grid[j,i]
                if cell == self.tree or cell == self.empty:
                    # To make the set space a bit smaller, the two are
                    # interechanged for tree
                    cell = self.tree
                s += str(cell)
        return s

    def ExpandGrid(self):
        # Function to artificially expand the grid for Same padding
        size = self.grid.shape
        PadGrid = np.zeros((size[0]+2,size[1]+2), dtype=np.int16)
        PadGrid[:,:] = self.empty
        PadGrid[1:-1,1:-1] = self.grid # Prop the original grid into the padded one
        return PadGrid
    