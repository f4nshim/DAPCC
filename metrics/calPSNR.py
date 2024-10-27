import open3d as o3d
import numpy as np 
import os, time
import pandas as pd
import subprocess
import glob
from tqdm import tqdm

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

def pc_error(infile1, infile2, res, exe_path, normal=False, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
               "h.       1(p2point)", "h.,PSNR  1(p2point)" ]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)", 
               "h.       2(p2point)", "h.,PSNR  2(p2point)" ]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)", 
               "h.        (p2point)", "h.,PSNR   (p2point)" ]

    haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF + haders_p2plane

    command = str(exe_path+'/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                          ' -n '+infile1+
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(res))

    if normal:
      headers += haders_p2plane
      command = str(command + ' -n ' + infile1)

    results = {}
   
    start = time.time()
    subp=subprocess.Popen(command, 
                          shell=True, stdout=subprocess.PIPE)

    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c = subp.stdout.readline()
    # print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

    return results


def compute_metrics_for_onefile(or_ply_path, rc_ply_path, exe_path, dataset_type):
    ori_pc = o3d.io.read_point_cloud(or_ply_path)
    ori_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30)) # compute normal
    o3d.io.write_point_cloud("./temp_pc_normal/temp.ply",ori_pc,write_ascii=True)
    # double to float
    lines = open("./temp_pc_normal/temp.ply").readlines()
    to_be_modified = [4,5,6,7,8,9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float')
    file = open("./temp_pc_normal/temp.ply", 'w')
    for line in lines:
        file.write(line)
    file.close()
    # cal metrics
    if dataset_type == 'KITTI':
        results = pc_error("./temp_pc_normal/temp.ply", rc_ply_path, res=59.7, exe_path=exe_path,normal=True)  # 59.7
    elif dataset_type == 'ScanNet':
        results = pc_error("./temp_pc_normal/temp.ply", rc_ply_path, res=1, exe_path=exe_path,normal=True)  # 59.7
    d1_psnr = results['mseF,PSNR (p2point)']
    d2_psnr = results['mseF,PSNR (p2plane)']
    d1_mse = results['mseF      (p2point)']
    res = {"d1_psnr":d1_psnr, "d2_psnr":d2_psnr, "d1_mse":d1_mse}
    return res

def compute_metrics_for_foler(or_ply_folder, rc_ply_foler, exe_path):
    d1_psnr_average = 0
    d2_psnr_average = 0
    d1_mse_average = 0
    or_files = glob.glob(or_ply_folder + "*.ply")
    rc_files = glob.glob(rc_ply_foler + "*.ply")
    or_files = sorted(or_files, key=lambda name: name)  
    rc_files = sorted(rc_files, key=lambda name: name)  
    for i in tqdm(range(len(or_files))):
        or_filename = or_files[i]
        rc_filename = rc_files[i]
        if(or_filename.split("/")[-1] != rc_filename.split("/")[-1]):
            print("filename not identical, please check")
            break
        result = compute_metrics_for_onefile(or_filename, rc_filename, exe_path)
        print("d1_psnr :{}, d2_psnr: {}".format(result['d1_psnr'],result['d2_psnr']))
        d1_psnr_average += result['d1_psnr']
        d2_psnr_average += result['d2_psnr']
        d1_mse_average += result['d1_mse']
    
    print("average d1_psnr {}, d2_psnr {}, mse {}".format(d1_psnr_average/len(or_files), d2_psnr_average/len(or_files), d1_mse_average/len(or_files)))
    return { "d1_psnr_avg": d1_psnr_average/len(or_files), 
             "d2_psnr_avg": d2_psnr_average/len(or_files),
             "d1_mse_avg": d1_mse_average/len(or_files)
            }

if __name__ == '__main__':
    # we set the root of octree as the first depth, while other methods set the root as the 0 depth
    # so here we show the reconstruct point cloud quality where the average bpp is 1.2716 in our paper.
    result = compute_metrics_for_onefile('../plys/KITTI/origin/13_000000.ply',
                                         '../plys/KITTI/13/1024/13_000000.ply', '.', 'KITTI')  
    print("d1_psnr :{}, d2_psnr: {}".format(result['d1_psnr'], result['d2_psnr']))