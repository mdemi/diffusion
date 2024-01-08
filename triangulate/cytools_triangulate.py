from cytools import *
import numpy as np
import pickle
import os, sys
from scipy.sparse import dok_matrix

# Parameters
height_images_dir = os.getcwd() + "/triangulate/samples/"
sample_out_dir = height_images_dir

# Load data
with open(height_images_dir + "height_images.pkl", "rb") as f:
    height_images = pickle.load(f)

# Construct triangulations
triangulations = []
for s in height_images:
    heights_image = s[0]
    heights_matrix = dok_matrix(heights_image)
    points = list(heights_matrix.keys())
    poly = Polytope(points)
    heights = np.array([heights_matrix.get(tuple(pt)) for pt in poly.points()])
    delaunay_heights = np.array([pt.dot(tuple(pt)) for pt in poly.points()])
    triangulation = poly.triangulate(heights, backend='qhull')
    triangulations.append(triangulation)
triangulations = np.array(triangulations)

points_list = [t.points() for t in triangulations]
simplices_list = [t.simplices() for t in triangulations]

# Save triangulations
with open(sample_out_dir + "points.pkl", "wb") as f:
    pickle.dump(points_list, f)
with open(sample_out_dir + "simplices.pkl", "wb") as f:
    pickle.dump(simplices_list, f)