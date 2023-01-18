import csv
from matplotlib import pyplot as plt
import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

import mpCustom as ct



dirName = sys.argv[1]
bootstrap_images_in_folder = f'./data/{dirName}/in'
bootstrap_images_out_folder = f'./data/{dirName}/out'
bootstrap_csvs_out_folder = f'./data/{dirName}/dist/'

bootstrap_helper = ct.BootstrapHelper(
    images_in_folder=bootstrap_images_in_folder,
    images_out_folder=bootstrap_images_out_folder,
    csvs_out_folder=bootstrap_csvs_out_folder,
)

bootstrap_helper.print_images_in_statistics()
bootstrap_helper.bootstrap(per_pose_class_limit=None)
bootstrap_helper.print_images_out_statistics()
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()

# os.listdir(
#     os.path.join('./data/down/')
# )
