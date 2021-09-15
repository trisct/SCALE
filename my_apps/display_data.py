import matplotlib.pyplot as plt
import argparse
import numpy as np
from my_utils.new_io_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('npzfilename', type=str)
args = parser.parse_args()

npzfilename = args.npzfilename

data = load_packed(npzfilename)
export_scalar_img_as_png(f'{npzfilename[:-4]}_uv.png', data['posmap32'])
export_points_as_xyz(f'{npzfilename[:-4]}_scan.xyz', data['scan_pc'], data['scan_n'])
export_points_as_xyz(f'{npzfilename[:-4]}_smpl.xyz', data['body_verts'])