import os

folder_path = '/home/ubelix/ana/khoma/MIALab2020/MIALab/experiments/'

for file in os.listdir(folder_path):
    os.system(f'sbatch {folder_path+file}')