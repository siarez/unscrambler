import numpy as np
from sklearn import datasets
import pickle
from scipy import signal
from openTSNE import TSNE as oTSNE
from openTSNE import  affinity
import math
import cv2
import os
import threading
import multiprocessing as mp
from threading import Timer
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull

import pygame, sys
from pygame.locals import *
import pygame.gfxdraw


def calc_dist(data):
    dist = 1 - np.abs(np.corrcoef(data, rowvar=True))
    # eliminating pixels with nan distance. These are pixels who have not changed across all images. That is std = 0
    np.nan_to_num(dist, copy=False)
    eleminated_pixel_idx = np.all(np.isnan(dist), axis=0)
    dist = dist[:, ~np.all(np.isnan(dist), axis=0)]
    dist = dist[~np.all(np.isnan(dist), axis=1), :]
    return dist, eleminated_pixel_idx

    
def my_voronoi_plot_2d(pixel_positions, color, DISPLAYSURF, **kw):
    """
    Plot the given Voronoi diagram in 2-D
    Parameters
    ----------
    pixel_positions : N,2 array
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points : bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points
    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot
    """   
    vor = Voronoi(pixel_positions)    

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    for point in range(vor.points.shape[0]):
        finite_segments = set()
        infinite_segments = set()
        ridge_points = [p for p in vor.ridge_points if point in p]
        ridge_vertices = [v for p, v in zip(vor.ridge_points, vor.ridge_vertices) if point in p]
        for pointidx, simplex in zip(ridge_points, ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                for i in simplex:                    
                    finite_segments.add(tuple(vor.vertices[i]))
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if (vor.furthest_site):
                    direction = -direction
                far_point = vor.vertices[i] + direction * ptp_bound.max()                
                infinite_segments.add(tuple(vor.vertices[i]))
                infinite_segments.add(tuple(far_point))

        polygon_points = np.array(list(finite_segments) + list(infinite_segments))
        polygon_points = polygon_points.clip(polygon_points.min() - 0.5, polygon_points.min() + 0.5)
        polygon_points = ((polygon_points - polygon_points.mean()) * 100) + 500  # shift and scale
        hull = ConvexHull(polygon_points)
        # print(polygon_points.min(), polygon_points.max())
        # ax.fill(polygon_points[hull.vertices, 0], polygon_points[hull.vertices, 1], color[point] )
        # pygame.draw.polygon(DISPLAYSURF, color[point], polygon_points[hull.vertices])
        pygame.gfxdraw.filled_polygon(DISPLAYSURF, polygon_points[hull.vertices], color[point])
        
        # dots = [[1,1], [2,1], [2,2], [1,2], [0.5,1.5]]
        # out = []
        # for x,y in dots:
        #     out.append([x*250, y*250])        
        # pygame.draw.polygon(DISPLAYSURF, (0, 255,0), out)            
        

    return


def plot_frame(frame, pixel_pos, DISPLAYSURF):
    frame_colors = frame.reshape(-1, 3)
    my_voronoi_plot_2d(pixel_pos, frame_colors[:], DISPLAYSURF)


def tsne_proc(q_send, q_receive):
    positions_embd = None
    while True: 
        if not q_receive.empty():
            frames_flat = q_receive.get()
            dist_mat, bad_pixel_idx = calc_dist(frames_flat)
            if dist_mat is None:
                q_send.put(None)
                break
            positions_embd = oTSNE(n_components=2, verbose=False, metric="precomputed", perplexity=10.0, \
                                   initialization='random' if positions_embd is None else positions_embd \
                                  ).prepare_initial(dist_mat)    
            positions_embd = positions_embd.optimize(n_iter=10, inplace=False, exaggeration=12 , n_jobs=-2) 
            q_send.put(positions_embd)
            print('Sent embeddings')
    print('BYYYEEEEE')
        
        
cap = cv2.VideoCapture(0)
# cap.set(3,64) # adjust width
# cap.set(4,48) # adjust height

# Need a running window of frames to calculate distances
window_size = 100
frames = []
positions_embd = None
positions_embd_avg = None
flip_flop_x = 0
flip_flop_y = 0

first_iter = True

ctx = mp.get_context('fork')    
q_receive = ctx.Queue()
q_send = ctx.Queue()
p = ctx.Process(target=tsne_proc, args=(q_receive, q_send))
p.start()


pygame.init()
DISPLAYSURF = pygame.display.set_mode((750, 750), 0, 32)
pygame.display.set_caption('WindowName')
DISPLAYSURF.fill((255,255,255))#< ; \/ - colours
print('Entering while loop')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can\'t receive frame')
        continue
    # frame = cv2.flip(frame, 1)[::24, ::24] # if your camera reverses your image
    frame = cv2.resize(frame, (frame.shape[0]//24, frame.shape[1]//24))
    # print(frame.shape)
    # flip_flop_x = 1 - flip_flop_x
    # flip_flop_y = 1 - flip_flop_y if flip_flop_x == 0 else flip_flop_y

    # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frames.append(frame)
    if len(frames) >= window_size:  
        if first_iter:
            frames_flat = np.stack(frames, axis=-1).reshape(-1, len(frames)*3)  
            dist_mat, bad_pixel_idx = calc_dist(frames_flat)
            q_send.put(dist_mat)
            first_iter = False
            positions_embd_avg = np.zeros((frame.shape[0] * frame.shape[1], 2))
        elif not q_receive.empty():
            frames_flat = np.stack(frames, axis=-1).reshape(-1, len(frames)*3)  
            # dist_mat, bad_pixel_idx = calc_dist(frames_flat) 
            q_send.put(frames_flat, block=False)
            positions_embd = q_receive.get()       

        if positions_embd is not None:
            mixing_coef = 0.95
            positions_embd_avg = positions_embd_avg*mixing_coef + positions_embd*(1-mixing_coef)
            plot_frame(frame, positions_embd_avg, DISPLAYSURF)
            print('drew polygon')
            pygame.display.update()
            
    if len(frames) >= 6 * window_size:
        frames.pop(0)


    for event in pygame.event.get():
        if event.type == QUIT:
            q_send.put(None, block=False)
            q_receive.get()  # waits for the other process to end
            pygame.quit()
            cap.release()
            p.join()
            sys.exit()    
  
# After the loop release the cap object
# cap.release()
# Destroy all the windows
# cv2.destroyAllWindows()
# p.terminate()
# p.join()