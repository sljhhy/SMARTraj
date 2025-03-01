import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import time
import pickle
from utils import Logger
import argparse
from task import road_cls, speed_inf, time_est, future_route_pre
from evluation_utils import get_road, fair_sampling, get_seq_emb_from_traj_withRouteOnly, get_seq_emb_from_traj_withALLModel, get_seq_emb_from_traj_withGridOnly, prepare_data
import torch
import os
torch.set_num_threads(5)

dev_id = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def evaluation(city, exp_path, model_name, start_time, sc):
    route_min_len, route_max_len, gps_min_len, gps_max_len, grid_min_len, grid_max_len = 10, 100, 10, 256, 6, 100
    model_path = os.path.join(exp_path, 'model', model_name)
    embedding_name = model_name.split('.')[0]

    # load task 1 & task2 label
    num_nodes = 10098
    print("num_nodes:", num_nodes)

    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model'] # model.to()包含inplace操作，不需要对象承接
    seq_model.grid_vocab_size = 9365 # chengdu
    seq_model.eval()

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")
# prepare sequence task
    test_seq_data = pickle.load(
        open('{}/{}_eval.pkl'.format(city, city),
             'rb'))
    test_seq_data = test_seq_data.sample(30000, random_state=219)

    route_length = test_seq_data['route_length'].values


    df = test_seq_data
    k = 1  # 设置k的值
    df_processed = df.apply(lambda row: process_row(row, k), axis=1)


    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, \
            grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat, \
            gps_length, gps_length_grid, dataset, mat_padding_value, mat_padding_value_grid = prepare_data(
        df_processed, route_min_len, route_max_len, gps_min_len, gps_max_len, grid_min_len, grid_max_len)


    seq_model.vocab_size = mat_padding_value # 
    seq_model.grid_vocab_size = mat_padding_value_grid # 
    seq_model.vocab_size_extra = mat_padding_value # 
    seq_model.grid_vocab_size_extra = mat_padding_value_grid # 
    route_data[:, :, 1] = torch.zeros_like(route_data[:, :, 1]).long()
    route_data[:, :, 2] = torch.full_like(route_data[:, :, 2], fill_value=-1)
    # test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, \
            grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat, \
            gps_length, gps_length_grid, dataset)
    seq_embedding = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=512, source_city=sc)
    # seq_embedding = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=900, source_city=sc)

    print("acc@1:")
    print("============")
    future_route_pre.evaluation(seq_embedding, dataset, num_nodes, k, acc_at=1)
    print("acc@5:")
    print("============")
    future_route_pre.evaluation(seq_embedding, dataset, num_nodes, k, acc_at=5)
    


def process_row(row, k):
    # 获取原始数据
    cpath_list = row['cpath_list']
    road_timestamp = row['road_timestamp']
    tm_list = row['tm_list']
    lat_list = row['lat_list']
    lng_list = row['lng_list']
    grid_timestamp = row['grid_timestamp']
    cgrid_list = row['cgrid_list']
    opath_list = row['opath_list']
    ogrid_list = row['ogrid_list']
    acceleration = row['acceleration']
    speed = row['speed']
    angle_delta = row['angle_delta']
    interval = row['interval']
    dist = row['dist']
    grid_fea = row['grid_fea']
    road_interval = row['road_interval']
    grid_interval = row['grid_interval']

    # 删除cgrid_list和grid_timestamp的最后k项
    pre_label = cgrid_list[-k:]
    cgrid_list = cgrid_list[:-k]
    grid_timestamp = grid_timestamp[:-k]
    grid_interval = grid_interval[:-k]
    grid_fea = grid_fea[:-k]

    # 获取grid_timestamp的最后一个值
    t = grid_timestamp[-1] if len(grid_timestamp) > 0 else None

    # 删除tm_list中大于t的项
    if t is not None:
        tm_list = [tm for tm in tm_list if tm <= t]
        num_deleted_tm = len([tm for tm in row['tm_list'] if tm > t])
    else:
        num_deleted_tm = 0

    # 删除lat_list和lng_list中与tm_list相同数量的项
    lat_list = lat_list[:len(tm_list)]
    lng_list = lng_list[:len(tm_list)]
    opath_list = opath_list[:len(tm_list)]
    ogrid_list = ogrid_list[:len(tm_list)]
    acceleration = acceleration[:len(tm_list)]
    speed = speed[:len(tm_list)]
    angle_delta = angle_delta[:len(tm_list)]
    interval = interval[:len(tm_list)]
    dist = dist[:len(tm_list)]

    # 删除road_timestamp中大于t的项
    if t is not None:
        road_timestamp = [gt for gt in road_timestamp if gt <= t]
        cpath_list = cpath_list[:len(road_timestamp)-1]
        road_interval = road_interval[:len(road_timestamp)-1]
        # num_deleted_grid = len([gt for gt in row['grid_timestamp'] if gt > t])
    # else:
    #     num_deleted_grid = 0

    return pd.Series({
        'cpath_list': cpath_list,
        'pre_label': pre_label,
        'road_timestamp': road_timestamp,
        'tm_list': tm_list,
        'lat_list': lat_list,
        'lng_list': lng_list,
        'grid_timestamp': grid_timestamp,
        'cgrid_list': cgrid_list,
        'num_deleted_tm': num_deleted_tm,
        # 'num_deleted_grid': num_deleted_grid,
        'opath_list': opath_list,
        'ogrid_list': ogrid_list,
        'acceleration': acceleration,
        'speed': speed,
        'angle_delta': angle_delta,
        'interval': interval,
        'dist': dist,
        'grid_fea': grid_fea,
        'road_interval': road_interval,
        'grid_interval': grid_interval
    })

if __name__ == '__main__':


    city, source_city = 'xian', True

    exp_path = 'research/exp/JTMR_xian_250128122816'
    model_name = 'JTMR_xian_v1_70_98900_250128122816_69.pt'

    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    evaluation(city, exp_path, model_name, start_time, source_city)