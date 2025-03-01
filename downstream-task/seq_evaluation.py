import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import time
import pickle
from utils import Logger
import argparse
from task import time_est
from evluation_utils import get_road, fair_sampling, get_seq_emb_from_traj_withRouteOnly, get_seq_emb_from_traj_withALLModel, prepare_data
import torch
import os
torch.set_num_threads(5)

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def evaluation(city, exp_path, model_name, start_time, sc):
    route_min_len, route_max_len, gps_min_len, gps_max_len, grid_min_len, grid_max_len = 10, 100, 10, 256, 10, 100
    model_path = os.path.join(exp_path, 'model', model_name)
    embedding_name = model_name.split('.')[0]

    # load task 1 & task2 label
    feature_df = pd.read_csv("{}/edge_features.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model'] # model.to()包含inplace操作，不需要对象承接
    seq_model.eval()

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare sequence task
    test_seq_data = pickle.load(
        open('{}/{}_eval.pkl'.format(city, city),'rb'))
    test_seq_data = test_seq_data.sample(50000, random_state=0)

    route_length = test_seq_data['route_length'].values
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, \
            grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat, \
            gps_length, gps_length_grid, dataset, mat_padding_value, mat_padding_value_grid = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len, grid_min_len, grid_max_len)
    
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
    seq_embedding = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=8, source_city=sc)

    time_est.evaluation(seq_embedding, dataset, num_nodes)


    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))

if __name__ == '__main__':

    city, source_city = 'xian', True
    exp_path = 'research/exp/JTMR_xian_250203092640'
    model_name = 'JTMR_xian_v1_70_98900_250203092640_65.pt'

    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    evaluation(city, exp_path, model_name, start_time, source_city)








