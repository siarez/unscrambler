import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
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
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

plt.switch_backend('Agg')

def calc_dist(data):
    dist = 1 - np.abs(np.corrcoef(data, rowvar=True))
    # eliminating pixels with nan distance. These are pixels who have not changed across all images. That is std = 0
    np.nan_to_num(dist, copy=False)
    eleminated_pixel_idx = np.all(np.isnan(dist), axis=0)
    dist = dist[:, ~np.all(np.isnan(dist), axis=0)]
    dist = dist[~np.all(np.isnan(dist), axis=1), :]
    return dist, eleminated_pixel_idx

def _adjust_bounds(ax, points):
    margin = 0.1 * points.ptp(axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])
    
def my_voronoi_plot_2d(pixel_positions, color, ax=None, **kw):
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

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], '.', markersize=point_size)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

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
        hull = ConvexHull(polygon_points)
        ax.fill(polygon_points[hull.vertices, 0], polygon_points[hull.vertices, 1], c=color[point])
    _adjust_bounds(ax, vor.points)

    return


def plot_frame(frame, pixel_pos, draw_lines_between_pixels=False):
    fig, ax = plt.subplots(1, 1, sharey=False, figsize=(2, 2), dpi=300)
    
    if draw_lines_between_pixels:
        lines = []
        pixel_pos_reshaped = pixel_pos.reshape(list(frame.shape[:2])+[2])
        for i in range(frame.shape[0] - 1):
            for j in range(frame.shape[1] - 1):
                point = pixel_pos_reshaped[i, j]
                right_neigh = pixel_pos_reshaped[i+1, j]
                bottom_neigh = pixel_pos_reshaped[i, j+1]
                lines.append((list(point), list(right_neigh)))
                lines.append((list(point), list(bottom_neigh)))

        lc = mc.LineCollection(lines, colors='grey', linewidths=0.4, alpha=0.5, zorder=1, linestyle='dashed')
        ax.add_collection(lc)        
    
    # ax.scatter(positions_embd[:, 0], positions_embd[:, 1], s=4, c=test_image[~bad_pixel_idx])
    frame_colors = frame.reshape(-1, 3) / 255 
    # print('colors shape', frame_colors.shape)
    ## ax.scatter(pixel_pos[~bad_pixel_idx, 0], pixel_pos[~bad_pixel_idx, 1], s=4, c=frame_colors[~bad_pixel_idx], alpha=0.5)
    
    ax.scatter(pixel_pos[:, 0], pixel_pos[:, 1], s=8, c=frame_colors[:], alpha=0.5)
    print(pixel_pos.min(), pixel_pos.max())
    # my_voronoi_plot_2d(pixel_pos, ax=ax, color=frame_colors[:], show_points=False, show_vertices=False)
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-4, 4)
    ax.set_facecolor("black")

    fig.canvas.draw()
    plot_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close()
    return plot_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


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
            plotted_frame = plot_frame(frame, positions_embd_avg)
            cv2.imshow('frame', plotted_frame)
            
    if len(frames) >= 6 * window_size:
        frames.pop(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        q_send.put(None, block=False)
        q_receive.get()  # waits for the other process to end
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
# p.terminate()
p.join()