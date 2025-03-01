import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from utils import weight_init
from dataloader import get_train_loader, random_mask
from utils import setup_seed
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from SMARTraj import SMARTraj
from cl_loss import get_traj_cl_loss, get_road_cl_loss, get_traj_cluster_loss, get_traj_match_loss, get_diff_loss
import os

dev_id = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)
torch.set_num_threads(10)

def train(config, config_extra):

    vocab_size_extra = config_extra['vocab_size']
    grid_vocab_size_extra = config_extra['grid_vocab_size']
    adj_path_extra = config_extra['adj_path']
    city_extra = config_extra['city']
    data_path_extra = config_extra['data_path']
    num_samples_extra = config_extra['num_samples']

    city = config['city']

    vocab_size = config['vocab_size']
    grid_vocab_size = config['grid_vocab_size']

    num_samples = config['num_samples']
    data_path = config['data_path']
    adj_path = config['adj_path']
    retrain = config['retrain']
    checkpoint_path = config['checkpoint_path']
    save_path = config['save_path']

    num_worker = config['num_worker']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    warmup_step = config['warmup_step']
    weight_decay = config['weight_decay']

    route_min_len = config['route_min_len']
    route_max_len = config['route_max_len']
    gps_min_len = config['gps_min_len']
    gps_max_len = config['gps_max_len']
    grid_min_len = config['grid_min_len']
    grid_max_len = config['grid_max_len']

    road_feat_num = config['road_feat_num']
    road_embed_size = config['road_embed_size']
    gps_feat_num = config['gps_feat_num']
    gps_embed_size = config['gps_embed_size']
    route_embed_size = config['route_embed_size']

    hidden_size = config['hidden_size']
    drop_route_rate = config['drop_route_rate'] # route_encoder
    drop_edge_rate = config['drop_edge_rate']   # gat
    drop_road_rate = config['drop_road_rate']   # sharedtransformer

    verbose = config['verbose']
    version = config['version']
    seed = config['random_seed']

    mask_length = config['mask_length']
    mask_prob = config['mask_prob']

    # 设置随机种子
    setup_seed(seed)

    # define model, parmeters and optimizer
    edge_index = np.load(adj_path)
    edge_index_extra = np.load(adj_path_extra)
    model = SMARTraj(vocab_size, grid_vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num,
                    gps_embed_size, route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate, drop_road_rate,
                    vocab_size_extra, grid_vocab_size_extra , edge_index_extra, mode='p', mode_extra='p').cuda()
    init_road_emb = torch.load('{}/init_w2v_road_emb.pt'.format(city), map_location='cuda:{}'.format(dev_id))
    model.node_embedding.weight = torch.nn.Parameter(init_road_emb['init_road_embd'])
    model.node_embedding.requires_grad_(True)
    init_road_emb_extra = torch.load('{}/init_w2v_road_emb.pt'.format(city_extra), map_location='cuda:{}'.format(dev_id))
    model.node_embedding_extra.weight = torch.nn.Parameter(init_road_emb_extra['init_road_embd'])
    model.node_embedding_extra.requires_grad_(True)
    print('load parameters in device {}'.format(model.node_embedding.weight.device)) # check process device

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # exp information
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    model_name = 'JTMR_{}_{}_{}_{}_{}'.format(city, version, num_epochs, num_samples, nowtime)
    model_path = os.path.join(save_path, 'JTMR_{}_{}'.format(city, nowtime), 'model')
    log_path = os.path.join(save_path, 'JTMR_{}_{}'.format(city, nowtime), 'log')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    writer = SummaryWriter(log_path)
    # if not retrain and checkpoints:
    if not retrain :
        checkpoint_path = '/research/exp/JTMR_xian_241223151244/model/JTMR_xian_v1_70_98900_241223151244_10.pt'
        print('loading {}'.format(checkpoint_path))
        model = torch.load(checkpoint_path, map_location="cuda:{}".format(dev_id))['model']
    else:
        model.apply(weight_init)

    num_last = min(num_samples, num_samples_extra)
    train_loader = get_train_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, grid_min_len, grid_max_len, num_last, seed)
    train_loader_extra = get_train_loader(data_path_extra, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, grid_min_len, grid_max_len, num_last, seed)
    print('dataset is ready.')

    epoch_step = train_loader.dataset.route_data.shape[0] // batch_size
    total_steps = epoch_step * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        for idx, (batch_sourse, batch_extra) in enumerate(zip(train_loader, train_loader_extra)):
            for batch, city_this in [(batch_sourse, city), (batch_extra, city_extra)]:
                gps_data, gps_assign_mat, route_data, route_assign_mat, \
                gps_data_grid, gps_assign_mat_grid, grid_data, grid_assign_mat, \
                gps_length, gps_length_grid = batch
                if city_this == city:
                    sourse_city = True
                    masked_route_assign_mat, masked_gps_assign_mat = random_mask(gps_assign_mat, route_assign_mat, gps_length,
                                                                            vocab_size, mask_length, mask_prob)
                    masked_grid_assign_mat, masked_gps_assign_mat_grid = random_mask(gps_assign_mat_grid, grid_assign_mat, gps_length_grid,
                                                                            grid_vocab_size, mask_length, mask_prob)
                else:
                    sourse_city = False
                    masked_route_assign_mat, masked_gps_assign_mat = random_mask(gps_assign_mat, route_assign_mat, gps_length,
                                                                            vocab_size_extra, mask_length, mask_prob)
                    masked_grid_assign_mat, masked_gps_assign_mat_grid = random_mask(gps_assign_mat_grid, grid_assign_mat, gps_length_grid,
                                                                            grid_vocab_size_extra, mask_length, mask_prob)
                gps_data, gps_assign_mat, route_data, route_assign_mat, \
                gps_data_grid, gps_assign_mat_grid, grid_data, grid_assign_mat, \
                gps_length, gps_length_grid = batch


                route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length =\
                    route_data.cuda(), masked_route_assign_mat.cuda(), gps_data.cuda(), masked_gps_assign_mat.cuda(), route_assign_mat.cuda(), gps_length.cuda()
                grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat, gps_length_grid = \
                    grid_data.cuda(), masked_grid_assign_mat.cuda(), gps_data_grid.cuda(), masked_gps_assign_mat_grid.cuda(), grid_assign_mat.cuda(), gps_length_grid.cuda()


                # route-gps 4个 、、grid-gps 4个 、、 过joint后的 head表示，  MLM表示
                gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
                gps_grid_rep, gps_traj_grid_rep, grid_road_rep, grid_traj_rep, \
                gps_traj_joint_rep, route_traj_joint_rep, gps_traj_grid_joint_rep, grid_traj_joint_rep, \
                gps_road_joint_rep, route_road_joint_rep, gps_grid_joint_rep, grid_road_joint_rep, \
                gps_traj_joint_rep_share, route_traj_joint_rep_share, gps_traj_grid_joint_rep_share, grid_traj_joint_rep_share, \
                gps_road_joint_rep_share, route_road_joint_rep_share, gps_grid_joint_rep_share, grid_road_joint_rep_share \
                    = model(route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, \
                            grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat, gps_length_grid, sourse_city)
                

                # DiffLoss 
                # pri_rep = torch.cat([temp_gps_road_joint_rep, temp_route_road_joint_rep, temp_gps_grid_joint_rep, temp_grid_road_joint_rep], dim=1)
                # share_rep = torch.cat([temp_gps_road_joint_rep_share, temp_route_road_joint_rep_share, temp_gps_grid_joint_rep_share, temp_grid_road_joint_rep_share], dim=1)
                pri_rep = torch.cat([gps_traj_joint_rep, route_traj_joint_rep, gps_traj_grid_joint_rep, grid_traj_joint_rep], dim=1)
                share_rep = torch.cat([gps_traj_joint_rep_share, route_traj_joint_rep_share, gps_traj_grid_joint_rep_share, grid_traj_joint_rep_share], dim=1)

                diff_loss = get_diff_loss(pri_rep, share_rep)
                
                del gps_road_rep, route_road_rep, gps_traj_joint_rep, route_traj_joint_rep, gps_traj_grid_joint_rep, grid_traj_joint_rep, \
                gps_traj_joint_rep_share, route_traj_joint_rep_share, gps_traj_grid_joint_rep_share, grid_traj_joint_rep_share, gps_grid_rep, grid_road_rep
                torch.cuda.empty_cache()

                # flatten road_rep
                mat2flatten = {}
                y_label = []
                if city_this == city:
                    route_length = (route_assign_mat != model.vocab_size).int().sum(1)
                else:
                    route_length = (route_assign_mat != model.vocab_size_extra).int().sum(1)
                # gps_road_list, route_road_list, gps_road_joint_list, route_road_joint_list = [], [], [], []
                gps_road_joint_list, route_road_joint_list = [], []
                now_flatten_idx = 0
                for i, length in enumerate(route_length):
                    y_label.append(route_assign_mat[i, :length]) # route 和 gps mask 位置是一样的
                    # gps_road_list.append(gps_road_rep[i, :length])
                    # route_road_list.append(route_road_rep[i, :length])
                    gps_road_joint_list.append(gps_road_joint_rep[i, :length])
                    route_road_joint_list.append(route_road_joint_rep[i, :length])
                    for l in range(length):
                        mat2flatten[(i, l)] = now_flatten_idx
                        now_flatten_idx += 1

                y_label = torch.cat(y_label, dim=0)
                gps_road_joint_rep = torch.cat(gps_road_joint_list, dim=0)
                route_road_joint_rep = torch.cat(route_road_joint_list, dim=0)

                # flatten road_rep_share
                gps_road_joint_list_share, route_road_joint_list_share = [], []
                now_flatten_idx = 0
                for i, length in enumerate(route_length):
                    gps_road_joint_list_share.append(gps_road_joint_rep_share[i, :length])
                    route_road_joint_list_share.append(route_road_joint_rep_share[i, :length])
                    for l in range(length):
                        now_flatten_idx += 1
                gps_road_joint_rep_share = torch.cat(gps_road_joint_list_share, dim=0)
                route_road_joint_rep_share = torch.cat(route_road_joint_list_share, dim=0)


                # flatten grid_rep
                mat2flatten_grid = {}
                y_label_grid = []
                if city_this == city:
                    grid_length = (grid_assign_mat != model.grid_vocab_size).int().sum(1)
                else:
                    grid_length = (grid_assign_mat != model.grid_vocab_size_extra).int().sum(1)
                # gps_grid_list, grid_road_list, gps_grid_joint_list, grid_road_joint_list = [], [], [], []
                gps_grid_joint_list, grid_road_joint_list = [], []
                now_flatten_idx_grid = 0
                for i, length in enumerate(grid_length):
                    y_label_grid.append(grid_assign_mat[i, :length]) # grid 和 gps mask 位置是一样的
                    gps_grid_joint_list.append(gps_grid_joint_rep[i, :length])
                    grid_road_joint_list.append(grid_road_joint_rep[i, :length])
                    for l in range(length):
                        mat2flatten_grid[(i, l)] = now_flatten_idx_grid
                        now_flatten_idx_grid += 1

                y_label_grid = torch.cat(y_label_grid, dim=0)
                gps_grid_joint_rep = torch.cat(gps_grid_joint_list, dim=0)
                grid_road_joint_rep = torch.cat(grid_road_joint_list, dim=0)

                # flatten grid_rep share
                gps_grid_joint_list_share, grid_road_joint_list_share = [], []
                now_flatten_idx_grid = 0
                for i, length in enumerate(grid_length):
                    gps_grid_joint_list_share.append(gps_grid_joint_rep_share[i, :length])
                    grid_road_joint_list_share.append(grid_road_joint_rep_share[i, :length])
                    for l in range(length):
                        now_flatten_idx_grid += 1
                gps_grid_joint_rep_share = torch.cat(gps_grid_joint_list_share, dim=0)
                grid_road_joint_rep_share = torch.cat(grid_road_joint_list_share, dim=0)

                # project rep into the same space
                gps_traj_rep = model.gps_proj_head(gps_traj_rep)
                route_traj_rep = model.route_proj_head(route_traj_rep)

                # project rep into the same space
                gps_traj_grid_rep = model.gps_proj_grid_head(gps_traj_grid_rep)
                grid_traj_rep = model.grid_proj_head(grid_traj_rep)

                # (GRM LOSS) get gps & route rep matching loss
                tau = 0.07
                match_loss = get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size, tau)
                # (GRM LOSS) get gps & grid rep matching loss
                tau2 = 0.07
                match_loss2 = get_traj_match_loss(gps_traj_grid_rep, grid_traj_rep, model, batch_size, tau2)
                # (GRM LOSS) get gps1 & gps2 rep matching loss
                tau3 = 0.07
                match_loss3 = get_traj_match_loss(gps_traj_rep, gps_traj_grid_rep, model, batch_size, tau3)

                

                # prepare label and mask_pos
                masked_pos = torch.nonzero(route_assign_mat != masked_route_assign_mat)
                masked_pos = [mat2flatten[tuple(pos.tolist())] for pos in masked_pos]
                y_label = y_label[masked_pos].long()



                # (MLM 1 LOSS) get gps rep road loss
                if city_this == city:
                    gps_mlm_pred = model.gps_mlm_head(gps_road_joint_rep) # project head 也会被更新
                    gps_mlm_pred_share = model.gps_mlm_head(gps_road_joint_rep_share) # project head 也会被更新
                else:
                    gps_mlm_pred = model.gps_mlm_head_extra(gps_road_joint_rep) # project head 也会被更新
                    gps_mlm_pred_share = model.gps_mlm_head_extra(gps_road_joint_rep_share) # project head 也会被更新
                masked_gps_mlm_pred = gps_mlm_pred[masked_pos]
                masked_gps_mlm_pred_share = gps_mlm_pred_share[masked_pos]
                gps_mlm_loss = nn.CrossEntropyLoss()(masked_gps_mlm_pred, y_label)
                gps_mlm_loss_share = nn.CrossEntropyLoss()(masked_gps_mlm_pred_share, y_label)

                # (MLM 2 LOSS) get route rep road loss
                if city_this == city:
                    route_mlm_pred = model.route_mlm_head(route_road_joint_rep) # project head 也会被更新
                    route_mlm_pred_share = model.route_mlm_head(route_road_joint_rep_share) # project head 也会被更新
                else:
                    route_mlm_pred = model.route_mlm_head_extra(route_road_joint_rep) # project head 也会被更新
                    route_mlm_pred_share = model.route_mlm_head_extra(route_road_joint_rep_share) # project head 也会被更新
                masked_route_mlm_pred = route_mlm_pred[masked_pos]
                masked_route_mlm_pred_share = route_mlm_pred_share[masked_pos]
                route_mlm_loss = nn.CrossEntropyLoss()(masked_route_mlm_pred, y_label)
                route_mlm_loss_share = nn.CrossEntropyLoss()(masked_route_mlm_pred_share, y_label)

                # prepare label and mask_pos
                masked_pos_grid = torch.nonzero(grid_assign_mat != masked_grid_assign_mat)
                masked_pos_grid = [mat2flatten_grid[tuple(pos.tolist())] for pos in masked_pos_grid]
                y_label_grid = y_label_grid[masked_pos_grid].long()

                # (MLM 3 LOSS) get gps2 rep road loss
                if city_this == city:
                    gps_grid_mlm_pred = model.gps_grid_mlm_head(gps_grid_joint_rep) # project head 也会被更新
                    gps_grid_mlm_pred_share = model.gps_grid_mlm_head(gps_grid_joint_rep_share) # project head 也会被更新
                else:
                    gps_grid_mlm_pred = model.gps_grid_mlm_head_extra(gps_grid_joint_rep) # project head 也会被更新
                    gps_grid_mlm_pred_share = model.gps_grid_mlm_head_extra(gps_grid_joint_rep_share) # project head 也会被更新
                masked_gps_grid_mlm_pred = gps_grid_mlm_pred[masked_pos_grid]
                masked_gps_grid_mlm_pred_share = gps_grid_mlm_pred_share[masked_pos_grid]
                gps_grid_mlm_loss = nn.CrossEntropyLoss()(masked_gps_grid_mlm_pred, y_label_grid)
                gps_grid_mlm_loss_share = nn.CrossEntropyLoss()(masked_gps_grid_mlm_pred_share, y_label_grid)

                # (MLM 4 LOSS) get grid rep road loss
                if city_this == city:
                    grid_mlm_pred = model.grid_mlm_head(grid_road_joint_rep) # project head 也会被更新
                    grid_mlm_pred_share = model.grid_mlm_head(grid_road_joint_rep_share) # project head 也会被更新
                else:
                    grid_mlm_pred = model.grid_mlm_head_extra(grid_road_joint_rep) # project head 也会被更新
                    grid_mlm_pred_share = model.grid_mlm_head_extra(grid_road_joint_rep_share) # project head 也会被更新
                masked_grid_mlm_pred = grid_mlm_pred[masked_pos_grid]
                masked_grid_mlm_pred_share = grid_mlm_pred_share[masked_pos_grid]
                grid_mlm_loss = nn.CrossEntropyLoss()(masked_grid_mlm_pred, y_label_grid)
                grid_mlm_loss_share = nn.CrossEntropyLoss()(masked_grid_mlm_pred_share, y_label_grid)



                # # MLM 1 LOSS + MLM 2 LOSS + GRM LOSS
                loss = (2*route_mlm_loss + gps_mlm_loss + grid_mlm_loss + gps_grid_mlm_loss + \
                        2*match_loss + 2*match_loss2 + 2*match_loss3 + \
                        2*route_mlm_loss_share + gps_mlm_loss_share + grid_mlm_loss_share + gps_grid_mlm_loss_share + \
                        diff_loss) / 12
                # loss = (route_mlm_loss + gps_mlm_loss/(gps_mlm_loss/route_mlm_loss).detach() + grid_mlm_loss/(grid_mlm_loss/route_mlm_loss).detach() + gps_grid_mlm_loss/(gps_grid_mlm_loss/route_mlm_loss).detach() + \
                #         match_loss/(match_loss/route_mlm_loss).detach() + match_loss2/(match_loss2/route_mlm_loss).detach() + match_loss3/(match_loss3/route_mlm_loss).detach() + \
                #         route_mlm_loss_share/(route_mlm_loss_share/route_mlm_loss).detach() + gps_mlm_loss_share/(gps_mlm_loss_share/route_mlm_loss).detach() + grid_mlm_loss_share/(grid_mlm_loss_share/route_mlm_loss).detach() + gps_grid_mlm_loss_share/(gps_grid_mlm_loss_share/route_mlm_loss).detach() + \
                #         diff_loss/(diff_loss/route_mlm_loss).detach()) / 12
                # loss = (3*route_mlm_loss + 3*gps_mlm_loss + 3*grid_mlm_loss + 3*gps_grid_mlm_loss + \
                #         2*match_loss + 2*match_loss2 + 2*match_loss3 + \
                #         3*route_mlm_loss_share + 3*gps_mlm_loss_share + 3*grid_mlm_loss_share + 3*gps_grid_mlm_loss_share + \
                #         diff_loss) / 12

                step = epoch_step*epoch + idx
                writer.add_scalar('{}/match_loss/match_loss'.format(city_this), match_loss, step)
                writer.add_scalar('{}/mlm_loss/gps_mlm_loss'.format(city_this), gps_mlm_loss, step)
                writer.add_scalar('{}/mlm_loss/route_mlm_loss'.format(city_this), route_mlm_loss, step)
                writer.add_scalar('{}/match_loss/match_loss2'.format(city_this), match_loss2, step)
                writer.add_scalar('{}/mlm_loss/gps_grid_mlm_loss'.format(city_this), gps_grid_mlm_loss, step)
                writer.add_scalar('{}/mlm_loss/grid_mlm_loss'.format(city_this), grid_mlm_loss, step)
                writer.add_scalar('{}/match_loss/match_loss3'.format(city_this), match_loss3, step)
                writer.add_scalar('{}/mlm_loss_share/gps_mlm_loss_share'.format(city_this), gps_mlm_loss_share, step)
                writer.add_scalar('{}/mlm_loss_share/route_mlm_loss_share'.format(city_this), route_mlm_loss_share, step)
                writer.add_scalar('{}/mlm_loss_share/gps_grid_mlm_loss_share'.format(city_this), gps_grid_mlm_loss_share, step)
                writer.add_scalar('{}/mlm_loss_share/grid_mlm_loss_share'.format(city_this), grid_mlm_loss_share, step)

                # writer.add_scalar('{}/diff_loss'.format(city_this), diff_loss, step)

                writer.add_scalar('{}/loss'.format(city_this), loss, step)

                del route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, \
                grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat, gps_length_grid 
                torch.cuda.empty_cache()

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                if not (idx + 1) % verbose:
                    t = datetime.now().strftime('%m-%d %H:%M:%S')
                    print(f'{t} | (Train) | Epoch={epoch}\tbatch_id={idx + 1}\tloss={loss.item():.4f}')

        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_path, "_".join([model_name, f'{epoch}.pt'])))

    return model

if __name__ == '__main__':
    config = json.load(open('config/xian.json', 'r'))
    config_extra = json.load(open('config/chengdu.json', 'r'))
    train(config, config_extra)


