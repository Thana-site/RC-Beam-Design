import streamlit as st
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

def calculate_bottom_centroid(steel_list, diameters, H, covering, ds, bottom_row_spacing):
    """Calculate centroid for the bottom reinforcement bars."""
    total_area = 0
    total_moment = 0
    y_list = []

    for i, (num_bars, diameter) in enumerate(zip(steel_list, diameters)):
        As = math.pi * (0.5 * diameter) ** 2
        ys = covering + ds + (diameter / 2) + i * (diameter + bottom_row_spacing)
        moment = As * num_bars * ys

        total_area += As * num_bars
        total_moment += moment
        y_list.append(round(ys, 3))

    centroid = round(total_moment / total_area, 3) if total_area != 0 else 0
    return round(total_area, 3), round(total_moment, 3), y_list, centroid

def calculate_top_centroid(steel_list, diameters, H, covering, ds, top_row_spacing):
    """Calculate centroid for the top reinforcement bars."""
    total_area = 0
    total_moment = 0
    y_list = []

    for i, (num_bars, diameter) in enumerate(zip(steel_list, diameters)):
        As = math.pi * (0.5 * diameter) ** 2
        ys = H - (covering + ds + (diameter / 2) + i * (diameter + top_row_spacing))
        moment = As * num_bars * ys

        total_area += As * num_bars
        total_moment += moment
        y_list.append(round(ys, 3))

    centroid = round(total_moment / total_area, 3) if total_area != 0 else 0
    return round(total_area, 3), round(total_moment, 3), y_list, centroid