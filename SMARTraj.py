import joblib
import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
from basemodel import BaseModel
import torch.nn.utils.rnn as rnn_utils

class SMARTraj(BaseModel):
    def __init__(self, vocab_size, grid_vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num, gps_embed_size, route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate, drop_road_rate, \
                 vocab_size_extra, grid_vocab_size_extra , edge_index_extra, mode='p', mode_extra='p'):
        super(SMARTraj, self).__init__()

        #base info
        self.vocab_size = vocab_size # 路段数量
        self.grid_vocab_size = grid_vocab_size # grid数量
        self.edge_index = torch.tensor(edge_index).cuda()
        self.mode = mode
        self.drop_edge_rate = drop_edge_rate
        #base info extra
        self.vocab_size_extra = vocab_size_extra # 路段数量
        self.grid_vocab_size_extra = grid_vocab_size_extra # grid数量
        self.edge_index_extra = torch.tensor(edge_index_extra).cuda()
        self.mode_extra = mode_extra

        # node embedding
        self.route_padding_vec = torch.zeros(1, road_embed_size, requires_grad=True).cuda()
        self.node_embedding = nn.Embedding(vocab_size, road_embed_size)
        self.node_embedding.requires_grad_(True)
        # node embedding extra
        self.route_padding_vec_extra = torch.zeros(1, road_embed_size, requires_grad=True).cuda()
        self.node_embedding_extra = nn.Embedding(vocab_size_extra, road_embed_size)
        self.node_embedding_extra.requires_grad_(True)

        # grid embedding
        self.grid_padding_vec = torch.zeros(1, road_embed_size, requires_grad=True).cuda()
        self.grid_embedding = nn.Embedding(grid_vocab_size, road_embed_size)
        self.grid_embedding.requires_grad_(True)
        # grid embedding extra
        self.grid_padding_vec_extra = torch.zeros(1, road_embed_size, requires_grad=True).cuda()
        self.grid_embedding_extra = nn.Embedding(grid_vocab_size_extra, road_embed_size)
        self.grid_embedding_extra.requires_grad_(True)

        # region embedding
        self.region_embedding = nn.Parameter(torch.randn(13, road_embed_size))

        # time embedding 考虑加法, 保证 embedding size一致
        self.minute_embedding = nn.Embedding(1440 + 1, route_embed_size)    # 0 是mask位
        self.week_embedding = nn.Embedding(7 + 1, route_embed_size)         # 0 是mask位
        self.delta_embedding = IntervalEmbedding(100, route_embed_size)     # -1 是mask位

        # route encoding
        self.graph_encoder = GraphEncoder(road_embed_size, route_embed_size)
        self.position_embedding1 = nn.Embedding(route_max_len, route_embed_size)
        self.fc1 = nn.Linear(route_embed_size, hidden_size) # route fuse time ffn
        self.route_encoder = TransformerModel(hidden_size, 8, hidden_size, 4, drop_route_rate)
        # route encoding extra
        self.graph_encoder_extra = GraphEncoder(road_embed_size, route_embed_size)
        self.position_embedding1_extra = nn.Embedding(route_max_len, route_embed_size)
        self.fc1_extra = nn.Linear(route_embed_size, hidden_size) # route fuse time ffn
        self.route_encoder_extra = TransformerModel(hidden_size, 8, hidden_size, 4, drop_edge_rate)

        # grid encoding
        self.position_embedding3 = nn.Embedding(route_max_len, route_embed_size)
        self.fc3 = nn.Linear(route_embed_size, hidden_size) # route fuse time ffn
        self.grid_encoder = TransformerModel(hidden_size, 8, hidden_size, 4, drop_route_rate)
        # grid encoding extra
        self.position_embedding3_extra = nn.Embedding(route_max_len, route_embed_size)
        self.fc3_extra = nn.Linear(route_embed_size, hidden_size) # route fuse time ffn
        self.grid_encoder_extra = TransformerModel(hidden_size, 8, hidden_size, 4, drop_route_rate)

        # gps encoding
        self.gps_linear = nn.Linear(gps_feat_num, gps_embed_size)
        self.gps_intra_encoder = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True) # 路段内建模
        self.gps_inter_encoder = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True) # 路段间建模

        # gps encoding grid
        self.gps_linear_grid = nn.Linear(gps_feat_num, gps_embed_size)
        self.gps_intra_encoder_grid = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True) # grid内建模
        self.gps_inter_encoder_grid = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True) # grid间建模

        # cl project head
        self.gps_proj_head = nn.Linear(2*gps_embed_size, hidden_size)
        self.route_proj_head = nn.Linear(hidden_size, hidden_size)

        self.gps_proj_grid_head = nn.Linear(2*gps_embed_size, hidden_size)
        self.grid_proj_head = nn.Linear(hidden_size, hidden_size) 


        # shared transformer
        self.position_embedding2 = nn.Embedding(route_max_len, hidden_size)
        self.modal_embedding = nn.Embedding(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) # shared transformer position transform
        self.fc4 = nn.Linear(hidden_size, hidden_size) # shared transformer position transform
        self.sharedtransformer = TransformerModel(hidden_size, 4, hidden_size, 2, drop_road_rate)
        # shared transformer extra
        self.position_embedding2_extra = nn.Embedding(route_max_len, hidden_size)
        self.fc2_extra = nn.Linear(hidden_size, hidden_size) # shared transformer position transform
        self.fc4_extra = nn.Linear(hidden_size, hidden_size) # shared transformer position transform
        self.sharedtransformer_extra = TransformerModel(hidden_size, 4, hidden_size, 2, drop_road_rate)
        # city shared transformer extra
        self.position_embedding2_share = nn.Embedding(route_max_len, hidden_size)
        self.fc2_share = nn.Linear(hidden_size, hidden_size) # shared transformer position transform
        self.fc4_share = nn.Linear(hidden_size, hidden_size) # shared transformer position transform
        self.sharedtransformer_share = TransformerModel(hidden_size, 4, hidden_size, 2, drop_road_rate)

        self.modal_embedding_four_module = nn.Embedding(4, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size) # shared transformer position transform

        # mlm classifier head
        self.gps_mlm_head = nn.Linear(hidden_size, vocab_size)
        self.route_mlm_head = nn.Linear(hidden_size, vocab_size)
        self.gps_grid_mlm_head = nn.Linear(hidden_size, grid_vocab_size)
        self.grid_mlm_head = nn.Linear(hidden_size, grid_vocab_size)
        # mlm classifier head extra
        self.gps_mlm_head_extra = nn.Linear(hidden_size, vocab_size_extra)
        self.route_mlm_head_extra = nn.Linear(hidden_size, vocab_size_extra)
        self.gps_grid_mlm_head_extra = nn.Linear(hidden_size, grid_vocab_size_extra)
        self.grid_mlm_head_extra = nn.Linear(hidden_size, grid_vocab_size_extra)

        # matching
        self.matching_predictor = nn.Linear(hidden_size*2, 2)
        # self.register_buffer("gps_queue", torch.randn(hidden_size, 2048))
        # self.register_buffer("route_queue", torch.randn(hidden_size, 2048))
        # self.image_queue = nn.functional.normalize(self.gps_queue, dim=0)
        # self.text_queue = nn.functional.normalize(self.route_queue, dim=0)


        self.add_cross = True
        #cross attention
        self.cross_attn_route_to_gps = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_route_to_grid = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_route_to_gps_g = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_route_to_route = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_route = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_grid = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_gps_g = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_gps_to_gps = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_route = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_gps = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_gps_g = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_grid_to_grid = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_route = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_gps = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_grid = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_gps_g_to_gps_g = CrossAttention(hidden_size, 8).cuda()

        #cross attention extra
        self.cross_attn_route_to_gps_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_route_to_grid_extra= CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_route_to_gps_g_extra = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_route_to_route_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_route_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_grid_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_gps_g_extra = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_gps_to_gps_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_route_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_gps_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_gps_g_extra = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_grid_to_grid_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_route_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_gps_extra = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_grid_extra = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_gps_g_to_gps_g_extra = CrossAttention(hidden_size, 8).cuda()

        #cross attention share
        self.cross_attn_route_to_gps_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_route_to_grid_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_route_to_gps_g_share = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_route_to_route_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_route_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_grid_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_to_gps_g_share = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_gps_to_gps_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_route_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_gps_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_grid_to_gps_g_share = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_grid_to_grid_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_route_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_gps_share = CrossAttention(hidden_size, 8).cuda()
        self.cross_attn_gps_g_to_grid_share = CrossAttention(hidden_size, 8).cuda()
        # self.cross_attn_gps_g_to_gps_g_share = CrossAttention(hidden_size, 8).cuda()

        self.add_pgmnet_city = True
        self.domain_linear = FeatureEncoder(25,256)
        self.pgmnet_city = PGMNet(hidden_size, hidden_size * 4)
        self.add_pgmnet_traj = True
        self.prior_linear = FeatureEncoder(256,256)
        self.pgmnet_traj = PGMNet(hidden_size, hidden_size * 4)
        self.add_pgmnet_traj_2 = False
        self.pgmnet_traj_2 = PGMNet(hidden_size, hidden_size * 4)



    def encode_graph(self, drop_rate=0.):
        node_emb = self.node_embedding.weight
        edge_index = dropout_adj(self.edge_index, p=drop_rate)[0]
        node_enc = self.graph_encoder(node_emb, edge_index)
        return node_enc

    def encode_route(self, route_data, route_assign_mat, masked_route_assign_mat):
        # 返回路段表示和轨迹的表示
        if self.mode == 'p':
            lookup_table = torch.cat([self.node_embedding.weight, self.route_padding_vec], 0)
        else:
            node_enc = self.encode_graph(self.drop_edge_rate)
            lookup_table = torch.cat([node_enc, self.route_padding_vec], 0)

        # 先对原始序列进行mask，然后再进行序列建模，防止信息泄露
        batch_size, max_seq_len = masked_route_assign_mat.size()

        src_key_padding_mask = (route_assign_mat == self.vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1) # 0 为padding位

        route_emb = torch.index_select(
            lookup_table, 0, masked_route_assign_mat.int().view(-1)).view(batch_size, max_seq_len, -1)

        # time embedding
        if route_data is None: # node evaluation的时候使用
            # 取时间表示的平均作为无时间特征输入时的表示
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            week_data = route_data[:, :, 0].long()
            min_data = route_data[:, :, 1].long()
            delta_data = route_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # position embedding
        position = torch.arange(route_emb.shape[1]).long().cuda()
        pos_emb = position.unsqueeze(0).repeat(route_emb.shape[0], 1)  # (S,) -> (B, S)
        pos_emb = self.position_embedding1(pos_emb)

        # fuse info
        route_emb = route_emb + pos_emb + week_emb + min_emb + delta_emb
        route_emb = self.fc1(route_emb)
        route_enc = self.route_encoder(route_emb, None, src_key_padding_mask) # mask 被在这里处理，mask不参与计算attention
        route_enc = torch.where(torch.isnan(route_enc), torch.full_like(route_enc, 0), route_enc) # 将nan变为0,防止溢出

        route_enc = route_enc.permute(1, 0, 2)  # 从 (47, 32, 256) 调整为 (32, 47, 256)
        route_unpooled = route_enc * pool_mask.repeat(1, 1, route_enc.shape[-1]) # (batch_size,max_len,feat_num)

        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        route_pooled = route_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1) # (batch_size, feat_num)

        return route_unpooled, route_pooled
    
    def encode_graph_extra(self, drop_rate=0.):
        node_emb = self.node_embedding_extra.weight
        edge_index = dropout_adj(self.edge_index_extra, p=drop_rate)[0]
        node_enc = self.graph_encoder_extra(node_emb, edge_index)
        return node_enc

    def encode_route_extra(self, route_data, route_assign_mat, masked_route_assign_mat):
        # 返回路段表示和轨迹的表示
        if self.mode == 'p':
            lookup_table = torch.cat([self.node_embedding_extra.weight, self.route_padding_vec_extra], 0)
        else:
            node_enc = self.encode_graph_extra(self.drop_edge_rate)
            lookup_table = torch.cat([node_enc, self.route_padding_vec_extra], 0)

        # 先对原始序列进行mask，然后再进行序列建模，防止信息泄露
        batch_size, max_seq_len = masked_route_assign_mat.size()

        src_key_padding_mask = (route_assign_mat == self.vocab_size_extra)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1) # 0 为padding位

        route_emb = torch.index_select(
            lookup_table, 0, masked_route_assign_mat.int().view(-1)).view(batch_size, max_seq_len, -1)

        # time embedding
        if route_data is None: # node evaluation的时候使用
            # 取时间表示的平均作为无时间特征输入时的表示
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            week_data = route_data[:, :, 0].long()
            min_data = route_data[:, :, 1].long()
            delta_data = route_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # position embedding
        position = torch.arange(route_emb.shape[1]).long().cuda()
        pos_emb = position.unsqueeze(0).repeat(route_emb.shape[0], 1)  # (S,) -> (B, S)
        pos_emb = self.position_embedding1_extra(pos_emb)

        # fuse info
        route_emb = route_emb + pos_emb + week_emb + min_emb + delta_emb
        route_emb = self.fc1_extra(route_emb)
        route_enc = self.route_encoder_extra(route_emb, None, src_key_padding_mask) # mask 被在这里处理，mask不参与计算attention
        route_enc = torch.where(torch.isnan(route_enc), torch.full_like(route_enc, 0), route_enc) # 将nan变为0,防止溢出

        route_enc = route_enc.permute(1, 0, 2)  # 从 (47, 32, 256) 调整为 (32, 47, 256)
        route_unpooled = route_enc * pool_mask.repeat(1, 1, route_enc.shape[-1]) # (batch_size,max_len,feat_num)

        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        route_pooled = route_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1) # (batch_size, feat_num)

        return route_unpooled, route_pooled
    
    def encode_grid(self, grid_data, grid_assign_mat, masked_grid_assign_mat):
        # 返回路段表示和轨迹的表示

        grid_embedding = self.grid_embedding.weight.data
        lookup_table = torch.cat([grid_embedding, self.grid_padding_vec], 0)  # grid的embedding

        # 先对原始序列进行mask，然后再进行序列建模，防止信息泄露
        batch_size, max_seq_len = masked_grid_assign_mat.size()

        src_key_padding_mask = (grid_assign_mat == self.grid_vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1) # 0 为padding位

        grid_emb = torch.index_select(
            lookup_table, 0, masked_grid_assign_mat.int().view(-1)).view(batch_size, max_seq_len, -1)

        # time embedding
        if grid_data is None: # node evaluation的时候使用
            # 取时间表示的平均作为无时间特征输入时的表示
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            week_data = grid_data[:, :, 0].long()
            min_data = grid_data[:, :, 1].long()
            delta_data = grid_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # regionn embedding
        weights = grid_data[:, :, 3:16]
        weighted_embeddings = torch.einsum('bmn,nd->bmd', weights, self.region_embedding)  # (batch_size, max_seq_len, embed_dim)


        grid_length = [length[length!=self.grid_vocab_size].shape[0]-1 for length in grid_assign_mat]
        first_embeddings = weighted_embeddings[:, 0, :]  # [batch_size, embedding_dim]
        last_embeddings = weighted_embeddings[torch.arange(weighted_embeddings.size(0)), grid_length, :]  # [batch_size, embedding_dim]
        prior_embedding = torch.cat([first_embeddings, last_embeddings], dim=-1)  # [batch_size, embedding_dim * 2]

        # position embedding
        position = torch.arange(grid_emb.shape[1]).long().cuda()
        pos_emb = position.unsqueeze(0).repeat(grid_emb.shape[0], 1)  # (S,) -> (B, S)
        pos_emb = self.position_embedding3(pos_emb)

        # fuse info
        grid_emb = grid_emb + pos_emb + week_emb + min_emb + delta_emb + weighted_embeddings
        grid_emb = self.fc3(grid_emb)
        grid_enc = self.grid_encoder(grid_emb, None, src_key_padding_mask) # mask 被在这里处理，mask不参与计算attention
        grid_enc = torch.where(torch.isnan(grid_enc), torch.full_like(grid_enc, 0), grid_enc) # 将nan变为0,防止溢出

        grid_enc = grid_enc.permute(1, 0, 2)  # 从 (47, 32, 256) 调整为 (32, 47, 256)
        grid_unpooled = grid_enc * pool_mask.repeat(1, 1, grid_enc.shape[-1]) # (batch_size,max_len,feat_num)

        # 对于单路段mask，可能存在整个grid都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        grid_pooled = grid_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1) # (batch_size, feat_num)

        return grid_unpooled, grid_pooled, prior_embedding

    def encode_grid_extra(self, grid_data, grid_assign_mat, masked_grid_assign_mat):
        # 返回路段表示和轨迹的表示

        grid_embedding = self.grid_embedding_extra.weight.data
        lookup_table = torch.cat([grid_embedding, self.grid_padding_vec_extra], 0)  # grid的embedding

        # 先对原始序列进行mask，然后再进行序列建模，防止信息泄露
        batch_size, max_seq_len = masked_grid_assign_mat.size()

        src_key_padding_mask = (grid_assign_mat == self.grid_vocab_size_extra)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1) # 0 为padding位

        grid_emb = torch.index_select(
            lookup_table, 0, masked_grid_assign_mat.int().view(-1)).view(batch_size, max_seq_len, -1)

        # time embedding
        if grid_data is None: # node evaluation的时候使用
            # 取时间表示的平均作为无时间特征输入时的表示
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            week_data = grid_data[:, :, 0].long()
            min_data = grid_data[:, :, 1].long()
            delta_data = grid_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # regionn embedding
        weights = grid_data[:, :, 3:16]
        weighted_embeddings = torch.einsum('bmn,nd->bmd', weights, self.region_embedding)  # (batch_size, max_seq_len, embed_dim)


        grid_length = [length[length!=self.grid_vocab_size_extra].shape[0]-1 for length in grid_assign_mat]
        first_embeddings = weighted_embeddings[:, 0, :]  # [batch_size, embedding_dim]
        last_embeddings = weighted_embeddings[torch.arange(weighted_embeddings.size(0)), grid_length, :]  # [batch_size, embedding_dim]
        prior_embedding = torch.cat([first_embeddings, last_embeddings], dim=-1)  # [batch_size, embedding_dim * 2]


        # position embedding
        position = torch.arange(grid_emb.shape[1]).long().cuda()
        pos_emb = position.unsqueeze(0).repeat(grid_emb.shape[0], 1)  # (S,) -> (B, S)
        pos_emb = self.position_embedding3_extra(pos_emb)

        # fuse info
        grid_emb = grid_emb + pos_emb + week_emb + min_emb + delta_emb + weighted_embeddings
        grid_emb = self.fc3_extra(grid_emb)
        grid_enc = self.grid_encoder_extra(grid_emb, None, src_key_padding_mask) # mask 被在这里处理，mask不参与计算attention
        grid_enc = torch.where(torch.isnan(grid_enc), torch.full_like(grid_enc, 0), grid_enc) # 将nan变为0,防止溢出

        grid_enc = grid_enc.permute(1, 0, 2)  # 从 (47, 32, 256) 调整为 (32, 47, 256)
        grid_unpooled = grid_enc * pool_mask.repeat(1, 1, grid_enc.shape[-1]) # (batch_size,max_len,feat_num)

        # 对于单路段mask，可能存在整个grid都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        grid_pooled = grid_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1) # (batch_size, feat_num)

        return grid_unpooled, grid_pooled, prior_embedding
    
    def encode_gps(self, gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length):
        # gps_data 先输入 gps_encoder, 输出每个step的output，选择路段对应位置的gps点的output进行pooling作为路段的表示
        gps_data = self.gps_linear(gps_data)

        # mask features
        gps_src_key_padding_mask = (masked_gps_assign_mat == self.vocab_size)
        gps_mask_mat = (1 - gps_src_key_padding_mask.int()).unsqueeze(-1).repeat(1, 1, gps_data.shape[-1]) # 0 为padding位
        masked_gps_data = gps_data * gps_mask_mat # (batch_size,gps_max_len,feat_num)

        # flatten gps data 便于进行路段内gru的并行
        flattened_gps_data, route_length = self.gps_flatten(masked_gps_data, gps_length) # flattened_gps_data (road_num, max_pt_len ,gps_fea_size)
        _, gps_emb = self.gps_intra_encoder(flattened_gps_data) # gps_emb (1, road_num, gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state
        gps_emb = gps_emb[-1] # 只保留前向的表示
        # gps_emb = torch.cat([gps_emb[0].squeeze(0), gps_emb[1].squeeze(0)],dim=-1) # 前后向表示拼接

        # stack gps emb 便于进行路段间gru的计算
        stacked_gps_emb = self.route_stack(gps_emb, route_length) # stacked_gps_emb (batch_size, max_route_len, gps_embed_size)
        gps_emb, _ = self.gps_inter_encoder(stacked_gps_emb)  # (batch_size, max_route_len, 2*gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state

        route_src_key_padding_mask = (masked_route_assign_mat == self.vocab_size).transpose(0, 1)
        route_pool_mask = (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1) # 包含mask的长度
        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(min=1) # mask 后的有值的路段数量，比路段长度要短
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled
    
    def encode_gps_extra(self, gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length):
        # gps_data 先输入 gps_encoder, 输出每个step的output，选择路段对应位置的gps点的output进行pooling作为路段的表示
        gps_data = self.gps_linear(gps_data)

        # mask features
        gps_src_key_padding_mask = (masked_gps_assign_mat == self.vocab_size_extra)
        gps_mask_mat = (1 - gps_src_key_padding_mask.int()).unsqueeze(-1).repeat(1, 1, gps_data.shape[-1]) # 0 为padding位
        masked_gps_data = gps_data * gps_mask_mat # (batch_size,gps_max_len,feat_num)

        # flatten gps data 便于进行路段内gru的并行
        flattened_gps_data, route_length = self.gps_flatten(masked_gps_data, gps_length) # flattened_gps_data (road_num, max_pt_len ,gps_fea_size)
        _, gps_emb = self.gps_intra_encoder(flattened_gps_data) # gps_emb (1, road_num, gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state
        gps_emb = gps_emb[-1] # 只保留前向的表示
        # gps_emb = torch.cat([gps_emb[0].squeeze(0), gps_emb[1].squeeze(0)],dim=-1) # 前后向表示拼接

        # stack gps emb 便于进行路段间gru的计算
        stacked_gps_emb = self.route_stack(gps_emb, route_length) # stacked_gps_emb (batch_size, max_route_len, gps_embed_size)
        gps_emb, _ = self.gps_inter_encoder(stacked_gps_emb)  # (batch_size, max_route_len, 2*gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state

        route_src_key_padding_mask = (masked_route_assign_mat == self.vocab_size_extra).transpose(0, 1)
        route_pool_mask = (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1) # 包含mask的长度
        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(min=1) # mask 后的有值的路段数量，比路段长度要短
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled

    def encode_gps_grid(self, gps_data_grid, masked_gps_assign_mat_grid, masked_grid_assign_mat, gps_length_grid):
        # gps_data 先输入 gps_encoder, 输出每个step的output，选择路段对应位置的gps点的output进行pooling作为路段的表示
        gps_data = self.gps_linear_grid(gps_data_grid)

        # mask features
        gps_src_key_padding_mask = (masked_gps_assign_mat_grid == self.grid_vocab_size)
        gps_mask_mat = (1 - gps_src_key_padding_mask.int()).unsqueeze(-1).repeat(1, 1, gps_data.shape[-1]) # 0 为padding位
        masked_gps_data = gps_data * gps_mask_mat # (batch_size,gps_max_len,feat_num)

        # flatten gps data 便于进行路段内gru的并行
        flattened_gps_data, route_length = self.gps_flatten(masked_gps_data, gps_length_grid) # flattened_gps_data (road_num, max_pt_len ,gps_fea_size)
        _, gps_emb = self.gps_intra_encoder_grid(flattened_gps_data) # gps_emb (1, road_num, gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state
        gps_emb = gps_emb[-1] # 只保留前向的表示

        # stack gps emb 便于进行路段间gru的计算
        stacked_gps_emb = self.route_stack(gps_emb, route_length) # stacked_gps_emb (batch_size, max_route_len, gps_embed_size)
        gps_emb, _ = self.gps_inter_encoder_grid(stacked_gps_emb)  # (batch_size, max_route_len, 2*gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state

        route_src_key_padding_mask = (masked_grid_assign_mat == self.grid_vocab_size).transpose(0, 1)
        route_pool_mask = (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1) # 包含mask的长度
        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(min=1) # mask 后的有值的路段数量，比路段长度要短
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled
    
    def encode_gps_grid_extra(self, gps_data_grid, masked_gps_assign_mat_grid, masked_grid_assign_mat, gps_length_grid):
        # gps_data 先输入 gps_encoder, 输出每个step的output，选择路段对应位置的gps点的output进行pooling作为路段的表示
        gps_data = self.gps_linear_grid(gps_data_grid)

        # mask features
        gps_src_key_padding_mask = (masked_gps_assign_mat_grid == self.grid_vocab_size_extra)
        gps_mask_mat = (1 - gps_src_key_padding_mask.int()).unsqueeze(-1).repeat(1, 1, gps_data.shape[-1]) # 0 为padding位
        masked_gps_data = gps_data * gps_mask_mat # (batch_size,gps_max_len,feat_num)

        # flatten gps data 便于进行路段内gru的并行
        flattened_gps_data, route_length = self.gps_flatten(masked_gps_data, gps_length_grid) # flattened_gps_data (road_num, max_pt_len ,gps_fea_size)
        _, gps_emb = self.gps_intra_encoder_grid(flattened_gps_data) # gps_emb (1, road_num, gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state
        gps_emb = gps_emb[-1] # 只保留前向的表示

        # stack gps emb 便于进行路段间gru的计算
        stacked_gps_emb = self.route_stack(gps_emb, route_length) # stacked_gps_emb (batch_size, max_route_len, gps_embed_size)
        gps_emb, _ = self.gps_inter_encoder_grid(stacked_gps_emb)  # (batch_size, max_route_len, 2*gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state

        route_src_key_padding_mask = (masked_grid_assign_mat == self.grid_vocab_size_extra).transpose(0, 1)
        route_pool_mask = (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1) # 包含mask的长度
        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(min=1) # mask 后的有值的路段数量，比路段长度要短
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled
    
    def route_stack(self, gps_emb, route_length):
        # flatten_gps_data tensor = (real_len, max_gps_in_route_len, emb_size)
        # route_length dict = { key:tid, value: road_len }
        values = list(route_length.values())
        route_max_len = max(values)
        data_list = []
        for idx in range(len(route_length)):
            start_idx = sum(values[:idx])
            end_idx = sum(values[:idx+1])
            data = gps_emb[start_idx:end_idx]
            data_list.append(data)

        stacked_gps_emb = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)

        return stacked_gps_emb

    def gps_flatten(self, gps_data, gps_length):
        # 把gps_data按照gps_assign_mat做形变，把每个路段上的gps点单独拿出来，拼成一个新的tensor (road_num, gps_max_len, gps_feat_num)，
        # 该tensor用于输入GRU进行并行计算
        traj_num, gps_max_len, gps_feat_num = gps_data.shape
        flattened_gps_list = []
        route_index = {}
        for idx in range(traj_num):
            gps_feat = gps_data[idx] # (max_len, feat_num)
            length_list = gps_length[idx] # (max_len, 1) [7,9,12,1,0,0,0,0,0,0] # padding_value = 0
            # 遍历每个轨迹中的路段
            for _idx, length in enumerate(length_list):
                if length != 0:
                    start_idx = sum(length_list[:_idx])
                    end_idx = start_idx + length_list[_idx]
                    cnt = route_index.get(idx, 0)
                    route_index[idx] = cnt+1
                    road_feat = gps_feat[start_idx:end_idx]
                    flattened_gps_list.append(road_feat)

        flattened_gps_data = rnn_utils.pad_sequence(flattened_gps_list, padding_value=0, batch_first=True) # (road_num, gps_max_len, gps_feat_num)


        return flattened_gps_data, route_index

    def encode_joint(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat):
        max_len = torch.max((route_assign_mat!=self.vocab_size).int().sum(1)).item()
        max_len = max_len*2+2
        data_list = []
        mask_list = []
        route_length = [length[length!=self.vocab_size].shape[0] for length in route_assign_mat]

        modal_emb0 = self.modal_embedding(torch.tensor(0).cuda())
        modal_emb1 = self.modal_embedding(torch.tensor(1).cuda())

        for i, length in enumerate(route_length):
            route_road_token = route_road_rep[i][:length]
            gps_road_token = gps_road_rep[i][:length]
            route_cls_token = route_traj_rep[i].unsqueeze(0)
            gps_cls_token = gps_traj_rep[i].unsqueeze(0)

            # position
            position = torch.arange(length+1).long().cuda()
            pos_emb = self.position_embedding2(position)

            # update route_emb
            route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
            modal_emb = modal_emb0.unsqueeze(0).repeat(length+1, 1)
            route_emb = route_emb + pos_emb + modal_emb
            route_emb = self.fc2(route_emb)

            # update gps_emb
            gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
            modal_emb = modal_emb1.unsqueeze(0).repeat(length+1, 1)
            gps_emb = gps_emb + pos_emb + modal_emb
            gps_emb = self.fc2(gps_emb)

            data = torch.cat([gps_emb, route_emb], dim=0)
            data_list.append(data)

            mask = torch.tensor([False] * data.shape[0]).cuda() # mask的位置为true
            mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        joint_emb = self.sharedtransformer(joint_data, None, mask_mat)
        joint_emb = joint_emb.permute(1, 0, 2)
        # 每一行的0 和 length+1 对应的是 gps_traj_rep 和 route_traj_rep
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack([joint_emb[i, length+1] for i, length in enumerate(route_length)], dim=0)

        gps_road_rep = rnn_utils.pad_sequence([joint_emb[i, 1:length+1] for i, length in enumerate(route_length)],
                                              padding_value=0, batch_first=True)
        route_road_rep = rnn_utils.pad_sequence([joint_emb[i, length+2:2*length+2] for i, length in enumerate(route_length)],
                                                padding_value=0, batch_first=True)

        return gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep
    
    def encode_joint_four_stream(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, \
                                 grid_road_rep, grid_traj_rep, gps_grid_rep, gps_traj_grid_rep, grid_assign_mat):
        
        max_len = torch.max((route_assign_mat!=self.vocab_size).int().sum(1)).item()
        max_len = max_len*2+2
        max_len_grid = torch.max((grid_assign_mat!=self.grid_vocab_size).int().sum(1)).item()
        max_len_grid = max_len_grid*2+2
        if not self.add_cross:
            data_list = []
            mask_list = []
            data_list_grid = []
            mask_list_grid = []
            route_length = [length[length!=self.vocab_size].shape[0] for length in route_assign_mat]
            grid_length = [length[length!=self.grid_vocab_size].shape[0] for length in grid_assign_mat]

            modal_emb0 = self.modal_embedding_four_module(torch.tensor(0).cuda())
            modal_emb1 = self.modal_embedding_four_module(torch.tensor(1).cuda())
            modal_emb2 = self.modal_embedding_four_module(torch.tensor(2).cuda())
            modal_emb3 = self.modal_embedding_four_module(torch.tensor(3).cuda())

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):

                route_road_token = route_road_rep[i][:length_r]
                gps_road_token = gps_road_rep[i][:length_r]
                route_cls_token = route_traj_rep[i].unsqueeze(0)
                gps_cls_token = gps_traj_rep[i].unsqueeze(0)

                grid_road_token = grid_road_rep[i][:length_g]
                gps_grid_token = gps_grid_rep[i][:length_g]
                grid_cls_token = grid_traj_rep[i].unsqueeze(0)
                gps_cls_grid_token = gps_traj_grid_rep[i].unsqueeze(0)

                # position
                position_r = torch.arange(length_r+1).long().cuda()
                pos_emb_r = self.position_embedding2(position_r)

                position_g = torch.arange(length_g+1).long().cuda()
                pos_emb_g = self.position_embedding2(position_g)

                # update route_emb
                route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
                modal_emb = modal_emb0.unsqueeze(0).repeat(length_r+1, 1)
                route_emb = route_emb + pos_emb_r + modal_emb
                route_emb = self.fc2(route_emb)

                # update grid_emb
                grid_emb = torch.cat([grid_cls_token, grid_road_token], dim=0)
                modal_emb = modal_emb2.unsqueeze(0).repeat(length_g+1, 1)
                grid_emb = grid_emb + pos_emb_g + modal_emb
                grid_emb = self.fc4(grid_emb)

                # update gps_emb_r
                gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
                modal_emb = modal_emb1.unsqueeze(0).repeat(length_r+1, 1)
                gps_emb = gps_emb + pos_emb_r + modal_emb
                gps_emb = self.fc2(gps_emb)

                # update gps_emb_g
                gps_g_emb = torch.cat([gps_cls_grid_token, gps_grid_token], dim=0)
                modal_emb = modal_emb3.unsqueeze(0).repeat(length_g+1, 1)
                gps_g_emb = gps_g_emb + pos_emb_g + modal_emb
                gps_g_emb = self.fc4(gps_g_emb)
                
                data = torch.cat([gps_emb, route_emb, gps_g_emb, grid_emb], dim=0)
                data_list.append(data)

                mask = torch.tensor([False] * data.shape[0]).cuda() # mask的位置为true
                mask_list.append(mask)
        else:
            data_list = []
            mask_list = []
            route_emb_list = []
            grid_emb_list = []
            gps_emb_list = []
            gps_g_emb_list = []

            mask_list_route_emb = []
            mask_list_grid_emb = []
            mask_list_gps_emb = []
            mask_list_gps_g_emb = []

            route_length = [length[length!=self.vocab_size].shape[0] for length in route_assign_mat]
            grid_length = [length[length!=self.grid_vocab_size].shape[0] for length in grid_assign_mat]

            modal_emb0 = self.modal_embedding_four_module(torch.tensor(0).cuda())
            modal_emb1 = self.modal_embedding_four_module(torch.tensor(1).cuda())
            modal_emb2 = self.modal_embedding_four_module(torch.tensor(2).cuda())
            modal_emb3 = self.modal_embedding_four_module(torch.tensor(3).cuda())

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):

                route_road_token = route_road_rep[i]
                gps_road_token = gps_road_rep[i]
                route_cls_token = route_traj_rep[i].unsqueeze(0)
                gps_cls_token = gps_traj_rep[i].unsqueeze(0)

                grid_road_token = grid_road_rep[i]
                gps_grid_token = gps_grid_rep[i]
                grid_cls_token = grid_traj_rep[i].unsqueeze(0)
                gps_cls_grid_token = gps_traj_grid_rep[i].unsqueeze(0)


                route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
                grid_emb = torch.cat([grid_cls_token, grid_road_token], dim=0)
                gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
                gps_g_emb = torch.cat([gps_cls_grid_token, gps_grid_token], dim=0)


                route_emb_list.append(route_emb)
                grid_emb_list.append(grid_emb)
                gps_emb_list.append(gps_emb)
                gps_g_emb_list.append(gps_g_emb)

                mask_route_emb = torch.tensor([False] * (length_r+1) + [True] * (route_emb.shape[0]-(length_r+1))).cuda()
                mask_list_route_emb.append(mask_route_emb)
                mask_grid_emb = torch.tensor([False] * (length_g+1) + [True] * (grid_emb.shape[0]-(length_g+1))).cuda()
                mask_list_grid_emb.append(mask_grid_emb)
                mask_gps_emb = torch.tensor([False] * (length_r+1) + [True] * (gps_emb.shape[0]-(length_r+1))).cuda()
                mask_list_gps_emb.append(mask_gps_emb)
                mask_gps_g_emb = torch.tensor([False] * (length_g+1) + [True] * (gps_g_emb.shape[0]-(length_g+1))).cuda()
                mask_list_gps_g_emb.append(mask_gps_g_emb)

            route_emb = torch.stack(route_emb_list, dim=0)
            mask_route_emb = torch.stack(mask_list_route_emb, dim=0)
            grid_emb = torch.stack(grid_emb_list, dim=0)
            mask_grid_emb = torch.stack(mask_list_grid_emb, dim=0)
            gps_emb = torch.stack(gps_emb_list, dim=0)
            mask_gps_emb = torch.stack(mask_list_gps_emb, dim=0)
            gps_g_emb = torch.stack(gps_g_emb_list, dim=0)
            mask_gps_g_emb = torch.stack(mask_list_gps_g_emb, dim=0)

            # 将输入张量的维度调整为 (S, N, E)，其中 S 是序列长度，N 是批次大小，E 是嵌入维度
            route_emb = route_emb.permute(1, 0, 2).cuda()
            gps_emb = gps_emb.permute(1, 0, 2).cuda()
            grid_emb = grid_emb.permute(1, 0, 2).cuda()
            gps_g_emb = gps_g_emb.permute(1, 0, 2).cuda()

            mask_route_emb = mask_route_emb.cuda()
            mask_grid_emb = mask_grid_emb.cuda()
            mask_gps_emb = mask_gps_emb.cuda()
            mask_gps_g_emb = mask_gps_g_emb.cuda()

            # Route embedding as query, others as key and value
            attn_route_to_gps = self.cross_attn_route_to_gps(route_emb, gps_emb, gps_emb,mask_route_emb, mask_gps_emb)
            attn_route_to_grid = self.cross_attn_route_to_grid(route_emb, grid_emb, grid_emb,mask_route_emb, mask_grid_emb)
            attn_route_to_gps_g = self.cross_attn_route_to_gps_g(route_emb, gps_g_emb, gps_g_emb,mask_route_emb, mask_gps_g_emb)
            # attn_route_to_route = self.cross_attn_route_to_route(route_emb, route_emb, route_emb,mask_route_emb, mask_route_emb)

            # GPS embedding as query, others as key and value
            attn_gps_to_route = self.cross_attn_gps_to_route(gps_emb, route_emb, route_emb,mask_gps_emb, mask_route_emb)
            attn_gps_to_grid = self.cross_attn_gps_to_grid(gps_emb, grid_emb, grid_emb,mask_gps_emb, mask_grid_emb)
            attn_gps_to_gps_g = self.cross_attn_gps_to_gps_g(gps_emb, gps_g_emb, gps_g_emb,mask_gps_emb, mask_gps_g_emb)
            # attn_gps_to_gps = self.cross_attn_gps_to_gps(gps_emb, gps_emb, gps_emb,mask_gps_emb, mask_gps_emb)

            # Grid embedding as query, others as key and value
            attn_grid_to_route = self.cross_attn_grid_to_route(grid_emb, route_emb, route_emb,mask_grid_emb, mask_route_emb)
            attn_grid_to_gps =self. cross_attn_grid_to_gps(grid_emb, gps_emb, gps_emb,mask_grid_emb, mask_gps_emb)
            attn_grid_to_gps_g = self.cross_attn_grid_to_gps_g(grid_emb, gps_g_emb, gps_g_emb,mask_grid_emb, mask_gps_g_emb)
            # attn_grid_to_grid = self.cross_attn_grid_to_grid(grid_emb, grid_emb, grid_emb,mask_grid_emb, mask_grid_emb)

            # GPS Grid embedding as query, others as key and value
            attn_gps_g_to_route = self.cross_attn_gps_g_to_route(gps_g_emb, route_emb, route_emb,mask_gps_g_emb, mask_route_emb)
            attn_gps_g_to_gps = self.cross_attn_gps_g_to_gps(gps_g_emb, gps_emb, gps_emb,mask_gps_g_emb, mask_gps_emb)
            attn_gps_g_to_grid = self.cross_attn_gps_g_to_grid(gps_g_emb, grid_emb, grid_emb,mask_gps_g_emb, mask_grid_emb)
            # attn_gps_g_to_gps_g = self.cross_attn_gps_g_to_gps_g(gps_g_emb, gps_g_emb, gps_g_emb,mask_gps_g_emb, mask_gps_g_emb)

            final_route_emb = attn_route_to_gps + attn_route_to_grid + attn_route_to_gps_g #+ attn_route_to_route
            final_gps_emb = attn_gps_to_route + attn_gps_to_grid + attn_gps_to_gps_g #+ attn_gps_to_gps
            final_grid_emb = attn_grid_to_route + attn_grid_to_gps + attn_grid_to_gps_g #+ attn_grid_to_grid
            final_gps_g_emb = attn_gps_g_to_route + attn_gps_g_to_gps + attn_gps_g_to_grid #+ attn_gps_g_to_gps_g

            final_route_emb = final_route_emb.permute(1, 0, 2)
            final_gps_emb = final_gps_emb.permute(1, 0, 2)
            final_grid_emb = final_grid_emb.permute(1, 0, 2)
            final_gps_g_emb = final_gps_g_emb.permute(1, 0, 2)

            non_route_padding_mask = ~mask_route_emb
            non_grid_padding_mask = ~mask_grid_emb
            non_gps_padding_mask = ~mask_gps_emb
            non_gps_g_padding_mask = ~mask_gps_g_emb

            # 对于每个 batch，使用非 padding mask 从 final_route_emb 中选择有效位置
            non_route_padding_emb = []
            non_grid_padding_emb = []
            non_gps_padding_emb = []
            non_gps_g_padding_emb = []

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):
                # 选择每个 batch 的非 padding 部分
                route_emb = final_route_emb[i][non_route_padding_mask[i]]
                non_route_padding_emb.append(route_emb)
                grid_emb = final_grid_emb[i][non_grid_padding_mask[i]]
                non_grid_padding_emb.append(grid_emb)
                gps_emb = final_gps_emb[i][non_gps_padding_mask[i]]
                non_gps_padding_emb.append(gps_emb)
                gps_g_emb = final_gps_g_emb[i][non_gps_g_padding_mask[i]]
                non_gps_g_padding_emb.append(gps_g_emb)


                # position
                position_r = torch.arange(length_r+1).long().cuda()
                pos_emb_r = self.position_embedding2(position_r) 

                position_g = torch.arange(length_g+1).long().cuda()
                pos_emb_g = self.position_embedding2(position_g) 

                # update route_emb
                modal_emb = modal_emb0.unsqueeze(0).repeat(length_r+1, 1)
                route_emb = route_emb + pos_emb_r + modal_emb
                route_emb = self.fc2(route_emb)

                # update grid_emb
                modal_emb = modal_emb2.unsqueeze(0).repeat(length_g+1, 1)
                grid_emb = grid_emb + pos_emb_g + modal_emb
                grid_emb = self.fc4(grid_emb)

                # update gps_emb_r
                modal_emb = modal_emb1.unsqueeze(0).repeat(length_r+1, 1)
                gps_emb = gps_emb + pos_emb_r + modal_emb
                gps_emb = self.fc2(gps_emb)

                # update gps_emb_g
                modal_emb = modal_emb3.unsqueeze(0).repeat(length_g+1, 1)
                gps_g_emb = gps_g_emb + pos_emb_g + modal_emb
                gps_g_emb = self.fc4(gps_g_emb)

                data = torch.cat([gps_emb, route_emb, gps_g_emb, grid_emb], dim=0)
                data_list.append(data)
                mask = torch.tensor([False] * data.shape[0]).cuda() # mask的位置为true
                mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        joint_emb = self.sharedtransformer(joint_data, None, mask_mat)
        joint_emb = joint_emb.permute(1, 0, 2)
        # 每一行的0 和 length+1 对应的是 gps_traj_rep 和 route_traj_rep
        # 2length+1 和 2length+length_g+1 对应的是 gps_traj_grid_rep 和 grid_traj_rep
        # 因为有cls
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack([joint_emb[i, length+1] for i, length in enumerate(route_length)], dim=0)
        gps_traj_grid_rep = torch.stack([joint_emb[i, 2*length+2] for i, length in enumerate(route_length)], dim=0)
        grid_traj_rep = torch.stack([joint_emb[i, 2*length_r+length_g+3] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))], dim=0)

        gps_road_rep = rnn_utils.pad_sequence([joint_emb[i, 1:length+1] for i, length in enumerate(route_length)],
                                              padding_value=0, batch_first=True)
        route_road_rep = rnn_utils.pad_sequence([joint_emb[i, length+2:2*length+2] for i, length in enumerate(route_length)],
                                                padding_value=0, batch_first=True)
        gps_grid_rep = rnn_utils.pad_sequence([joint_emb[i, 2*length_r+3:2*length_r+length_g+3] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))],
                                                padding_value=0, batch_first=True)
        grid_road_rep = rnn_utils.pad_sequence([joint_emb[i, 2*length_r+length_g+4:2*length_r+2*length_g+4] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))],
                                                padding_value=0, batch_first=True)

        return gps_traj_rep, route_traj_rep, gps_traj_grid_rep, grid_traj_rep, gps_road_rep, route_road_rep, gps_grid_rep, grid_road_rep
    

    def encode_joint_four_stream_extra(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, \
                                 grid_road_rep, grid_traj_rep, gps_grid_rep, gps_traj_grid_rep, grid_assign_mat):
        
        max_len = torch.max((route_assign_mat!=self.vocab_size_extra).int().sum(1)).item()
        max_len = max_len*2+2
        max_len_grid = torch.max((grid_assign_mat!=self.grid_vocab_size_extra).int().sum(1)).item()
        max_len_grid = max_len_grid*2+2
        if not self.add_cross:
            data_list = []
            mask_list = []
            data_list_grid = []
            mask_list_grid = []
            route_length = [length[length!=self.vocab_size_extra].shape[0] for length in route_assign_mat]
            grid_length = [length[length!=self.grid_vocab_size_extra].shape[0] for length in grid_assign_mat]

            modal_emb0 = self.modal_embedding_four_module(torch.tensor(0).cuda())
            modal_emb1 = self.modal_embedding_four_module(torch.tensor(1).cuda())
            modal_emb2 = self.modal_embedding_four_module(torch.tensor(2).cuda())
            modal_emb3 = self.modal_embedding_four_module(torch.tensor(3).cuda())

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):

                route_road_token = route_road_rep[i][:length_r]
                gps_road_token = gps_road_rep[i][:length_r]
                route_cls_token = route_traj_rep[i].unsqueeze(0)
                gps_cls_token = gps_traj_rep[i].unsqueeze(0)

                grid_road_token = grid_road_rep[i][:length_g]
                gps_grid_token = gps_grid_rep[i][:length_g]
                grid_cls_token = grid_traj_rep[i].unsqueeze(0)
                gps_cls_grid_token = gps_traj_grid_rep[i].unsqueeze(0)

                # position
                position_r = torch.arange(length_r+1).long().cuda()
                pos_emb_r = self.position_embedding2_extra(position_r)

                position_g = torch.arange(length_g+1).long().cuda()
                pos_emb_g = self.position_embedding2_extra(position_g)

                # update route_emb
                route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
                modal_emb = modal_emb0.unsqueeze(0).repeat(length_r+1, 1)
                route_emb = route_emb + pos_emb_r + modal_emb
                route_emb = self.fc2_extra(route_emb)

                # update grid_emb
                grid_emb = torch.cat([grid_cls_token, grid_road_token], dim=0)
                modal_emb = modal_emb2.unsqueeze(0).repeat(length_g+1, 1)
                grid_emb = grid_emb + pos_emb_g + modal_emb
                grid_emb = self.fc4_extra(grid_emb)

                # update gps_emb_r
                gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
                modal_emb = modal_emb1.unsqueeze(0).repeat(length_r+1, 1)
                gps_emb = gps_emb + pos_emb_r + modal_emb
                gps_emb = self.fc2_extra(gps_emb)

                # update gps_emb_g
                gps_g_emb = torch.cat([gps_cls_grid_token, gps_grid_token], dim=0)
                modal_emb = modal_emb3.unsqueeze(0).repeat(length_g+1, 1)
                gps_g_emb = gps_g_emb + pos_emb_g + modal_emb
                gps_g_emb = self.fc4_extra(gps_g_emb)
                
                data = torch.cat([gps_emb, route_emb, gps_g_emb, grid_emb], dim=0)
                data_list.append(data)

                mask = torch.tensor([False] * data.shape[0]).cuda() # mask的位置为true
                mask_list.append(mask)
        else:
            data_list = []
            mask_list = []
            route_emb_list = []
            grid_emb_list = []
            gps_emb_list = []
            gps_g_emb_list = []

            mask_list_route_emb = []
            mask_list_grid_emb = []
            mask_list_gps_emb = []
            mask_list_gps_g_emb = []

            route_length = [length[length!=self.vocab_size_extra].shape[0] for length in route_assign_mat]
            grid_length = [length[length!=self.grid_vocab_size_extra].shape[0] for length in grid_assign_mat]

            modal_emb0 = self.modal_embedding_four_module(torch.tensor(0).cuda())
            modal_emb1 = self.modal_embedding_four_module(torch.tensor(1).cuda())
            modal_emb2 = self.modal_embedding_four_module(torch.tensor(2).cuda())
            modal_emb3 = self.modal_embedding_four_module(torch.tensor(3).cuda())

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):

                route_road_token = route_road_rep[i]
                gps_road_token = gps_road_rep[i]
                route_cls_token = route_traj_rep[i].unsqueeze(0)
                gps_cls_token = gps_traj_rep[i].unsqueeze(0)

                grid_road_token = grid_road_rep[i]
                gps_grid_token = gps_grid_rep[i]
                grid_cls_token = grid_traj_rep[i].unsqueeze(0)
                gps_cls_grid_token = gps_traj_grid_rep[i].unsqueeze(0)


                route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
                grid_emb = torch.cat([grid_cls_token, grid_road_token], dim=0)
                gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
                gps_g_emb = torch.cat([gps_cls_grid_token, gps_grid_token], dim=0)


                route_emb_list.append(route_emb)
                grid_emb_list.append(grid_emb)
                gps_emb_list.append(gps_emb)
                gps_g_emb_list.append(gps_g_emb)

                mask_route_emb = torch.tensor([False] * (length_r+1) + [True] * (route_emb.shape[0]-(length_r+1))).cuda()
                mask_list_route_emb.append(mask_route_emb)
                mask_grid_emb = torch.tensor([False] * (length_g+1) + [True] * (grid_emb.shape[0]-(length_g+1))).cuda()
                mask_list_grid_emb.append(mask_grid_emb)
                mask_gps_emb = torch.tensor([False] * (length_r+1) + [True] * (gps_emb.shape[0]-(length_r+1))).cuda()
                mask_list_gps_emb.append(mask_gps_emb)
                mask_gps_g_emb = torch.tensor([False] * (length_g+1) + [True] * (gps_g_emb.shape[0]-(length_g+1))).cuda()
                mask_list_gps_g_emb.append(mask_gps_g_emb)

            route_emb = torch.stack(route_emb_list, dim=0)
            mask_route_emb = torch.stack(mask_list_route_emb, dim=0)
            grid_emb = torch.stack(grid_emb_list, dim=0)
            mask_grid_emb = torch.stack(mask_list_grid_emb, dim=0)
            gps_emb = torch.stack(gps_emb_list, dim=0)
            mask_gps_emb = torch.stack(mask_list_gps_emb, dim=0)
            gps_g_emb = torch.stack(gps_g_emb_list, dim=0)
            mask_gps_g_emb = torch.stack(mask_list_gps_g_emb, dim=0)

            # 将输入张量的维度调整为 (S, N, E)，其中 S 是序列长度，N 是批次大小，E 是嵌入维度
            route_emb = route_emb.permute(1, 0, 2).cuda()
            gps_emb = gps_emb.permute(1, 0, 2).cuda()
            grid_emb = grid_emb.permute(1, 0, 2).cuda()
            gps_g_emb = gps_g_emb.permute(1, 0, 2).cuda()

            mask_route_emb = mask_route_emb.cuda()
            mask_grid_emb = mask_grid_emb.cuda()
            mask_gps_emb = mask_gps_emb.cuda()
            mask_gps_g_emb = mask_gps_g_emb.cuda()

            # Route embedding as query, others as key and value
            attn_route_to_gps = self.cross_attn_route_to_gps_extra(route_emb, gps_emb, gps_emb,mask_route_emb, mask_gps_emb)
            attn_route_to_grid = self.cross_attn_route_to_grid_extra(route_emb, grid_emb, grid_emb,mask_route_emb, mask_grid_emb)
            attn_route_to_gps_g = self.cross_attn_route_to_gps_g_extra(route_emb, gps_g_emb, gps_g_emb,mask_route_emb, mask_gps_g_emb)
            # attn_route_to_route = self.cross_attn_route_to_route_extra(route_emb, route_emb, route_emb,mask_route_emb, mask_route_emb)

            # GPS embedding as query, others as key and value
            attn_gps_to_route = self.cross_attn_gps_to_route_extra(gps_emb, route_emb, route_emb,mask_gps_emb, mask_route_emb)
            attn_gps_to_grid = self.cross_attn_gps_to_grid_extra(gps_emb, grid_emb, grid_emb,mask_gps_emb, mask_grid_emb)
            attn_gps_to_gps_g = self.cross_attn_gps_to_gps_g_extra(gps_emb, gps_g_emb, gps_g_emb,mask_gps_emb, mask_gps_g_emb)
            # attn_gps_to_gps = self.cross_attn_gps_to_gps_extra(gps_emb, gps_emb, gps_emb,mask_gps_emb, mask_gps_emb)

            # Grid embedding as query, others as key and value
            attn_grid_to_route = self.cross_attn_grid_to_route_extra(grid_emb, route_emb, route_emb,mask_grid_emb, mask_route_emb)
            attn_grid_to_gps =self. cross_attn_grid_to_gps_extra(grid_emb, gps_emb, gps_emb,mask_grid_emb, mask_gps_emb)
            attn_grid_to_gps_g = self.cross_attn_grid_to_gps_g_extra(grid_emb, gps_g_emb, gps_g_emb,mask_grid_emb, mask_gps_g_emb)
            # attn_grid_to_grid = self.cross_attn_grid_to_grid_extra(grid_emb, grid_emb, grid_emb,mask_grid_emb, mask_grid_emb)

            # GPS Grid embedding as query, others as key and value
            attn_gps_g_to_route = self.cross_attn_gps_g_to_route_extra(gps_g_emb, route_emb, route_emb,mask_gps_g_emb, mask_route_emb)
            attn_gps_g_to_gps = self.cross_attn_gps_g_to_gps_extra(gps_g_emb, gps_emb, gps_emb,mask_gps_g_emb, mask_gps_emb)
            attn_gps_g_to_grid = self.cross_attn_gps_g_to_grid_extra(gps_g_emb, grid_emb, grid_emb,mask_gps_g_emb, mask_grid_emb)
            # attn_gps_g_to_gps_g = self.cross_attn_gps_g_to_gps_g_extra(gps_g_emb, gps_g_emb, gps_g_emb,mask_gps_g_emb, mask_gps_g_emb)

            final_route_emb = attn_route_to_gps + attn_route_to_grid + attn_route_to_gps_g #+ attn_route_to_route
            final_gps_emb = attn_gps_to_route + attn_gps_to_grid + attn_gps_to_gps_g #+ attn_gps_to_gps
            final_grid_emb = attn_grid_to_route + attn_grid_to_gps + attn_grid_to_gps_g #+ attn_grid_to_grid
            final_gps_g_emb = attn_gps_g_to_route + attn_gps_g_to_gps + attn_gps_g_to_grid #+ attn_gps_g_to_gps_g

            final_route_emb = final_route_emb.permute(1, 0, 2)
            final_gps_emb = final_gps_emb.permute(1, 0, 2)
            final_grid_emb = final_grid_emb.permute(1, 0, 2)
            final_gps_g_emb = final_gps_g_emb.permute(1, 0, 2)

            non_route_padding_mask = ~mask_route_emb
            non_grid_padding_mask = ~mask_grid_emb
            non_gps_padding_mask = ~mask_gps_emb
            non_gps_g_padding_mask = ~mask_gps_g_emb

            # 对于每个 batch，使用非 padding mask 从 final_route_emb 中选择有效位置
            non_route_padding_emb = []
            non_grid_padding_emb = []
            non_gps_padding_emb = []
            non_gps_g_padding_emb = []

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):
                # 选择每个 batch 的非 padding 部分
                route_emb = final_route_emb[i][non_route_padding_mask[i]]
                non_route_padding_emb.append(route_emb)
                grid_emb = final_grid_emb[i][non_grid_padding_mask[i]]
                non_grid_padding_emb.append(grid_emb)
                gps_emb = final_gps_emb[i][non_gps_padding_mask[i]]
                non_gps_padding_emb.append(gps_emb)
                gps_g_emb = final_gps_g_emb[i][non_gps_g_padding_mask[i]]
                non_gps_g_padding_emb.append(gps_g_emb)


                # position
                position_r = torch.arange(length_r+1).long().cuda()
                pos_emb_r = self.position_embedding2_extra(position_r) 

                position_g = torch.arange(length_g+1).long().cuda()
                pos_emb_g = self.position_embedding2_extra(position_g) 

                # update route_emb
                modal_emb = modal_emb0.unsqueeze(0).repeat(length_r+1, 1)
                route_emb = route_emb + pos_emb_r + modal_emb
                route_emb = self.fc2_extra(route_emb)

                # update grid_emb
                modal_emb = modal_emb2.unsqueeze(0).repeat(length_g+1, 1)
                grid_emb = grid_emb + pos_emb_g + modal_emb
                grid_emb = self.fc4_extra(grid_emb)

                # update gps_emb_r
                modal_emb = modal_emb1.unsqueeze(0).repeat(length_r+1, 1)
                gps_emb = gps_emb + pos_emb_r + modal_emb
                gps_emb = self.fc2_extra(gps_emb)

                # update gps_emb_g
                modal_emb = modal_emb3.unsqueeze(0).repeat(length_g+1, 1)
                gps_g_emb = gps_g_emb + pos_emb_g + modal_emb
                gps_g_emb = self.fc4_extra(gps_g_emb)

                data = torch.cat([gps_emb, route_emb, gps_g_emb, grid_emb], dim=0)
                data_list.append(data)
                mask = torch.tensor([False] * data.shape[0]).cuda() # mask的位置为true
                mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        joint_emb = self.sharedtransformer_extra(joint_data, None, mask_mat)
        joint_emb = joint_emb.permute(1, 0, 2)
        # 每一行的0 和 length+1 对应的是 gps_traj_rep 和 route_traj_rep
        # 2length+1 和 2length+length_g+1 对应的是 gps_traj_grid_rep 和 grid_traj_rep
        # 因为有cls
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack([joint_emb[i, length+1] for i, length in enumerate(route_length)], dim=0)
        gps_traj_grid_rep = torch.stack([joint_emb[i, 2*length+2] for i, length in enumerate(route_length)], dim=0)
        grid_traj_rep = torch.stack([joint_emb[i, 2*length_r+length_g+3] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))], dim=0)

        gps_road_rep = rnn_utils.pad_sequence([joint_emb[i, 1:length+1] for i, length in enumerate(route_length)],
                                              padding_value=0, batch_first=True)
        route_road_rep = rnn_utils.pad_sequence([joint_emb[i, length+2:2*length+2] for i, length in enumerate(route_length)],
                                                padding_value=0, batch_first=True)
        gps_grid_rep = rnn_utils.pad_sequence([joint_emb[i, 2*length_r+3:2*length_r+length_g+3] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))],
                                                padding_value=0, batch_first=True)
        grid_road_rep = rnn_utils.pad_sequence([joint_emb[i, 2*length_r+length_g+4:2*length_r+2*length_g+4] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))],
                                                padding_value=0, batch_first=True)

        return gps_traj_rep, route_traj_rep, gps_traj_grid_rep, grid_traj_rep, gps_road_rep, route_road_rep, gps_grid_rep, grid_road_rep

    def city_share_encode_joint(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, \
                                 grid_road_rep, grid_traj_rep, gps_grid_rep, gps_traj_grid_rep, grid_assign_mat, source_city):
        if source_city:
            vocab_this = self.vocab_size
            grid_vocab_this = self.grid_vocab_size
        else:
            vocab_this = self.vocab_size_extra
            grid_vocab_this = self.grid_vocab_size_extra
        max_len = torch.max((route_assign_mat!=vocab_this).int().sum(1)).item()
        max_len = max_len*2+2
        max_len_grid = torch.max((grid_assign_mat!=grid_vocab_this).int().sum(1)).item()
        max_len_grid = max_len_grid*2+2
        if not self.add_cross:
            data_list = []
            mask_list = []
            data_list_grid = []
            mask_list_grid = []
            route_length = [length[length!=vocab_this].shape[0] for length in route_assign_mat]
            grid_length = [length[length!=grid_vocab_this].shape[0] for length in grid_assign_mat]

            modal_emb0 = self.modal_embedding_four_module(torch.tensor(0).cuda())
            modal_emb1 = self.modal_embedding_four_module(torch.tensor(1).cuda())
            modal_emb2 = self.modal_embedding_four_module(torch.tensor(2).cuda())
            modal_emb3 = self.modal_embedding_four_module(torch.tensor(3).cuda())

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):

                route_road_token = route_road_rep[i][:length_r]
                gps_road_token = gps_road_rep[i][:length_r]
                route_cls_token = route_traj_rep[i].unsqueeze(0)
                gps_cls_token = gps_traj_rep[i].unsqueeze(0)

                grid_road_token = grid_road_rep[i][:length_g]
                gps_grid_token = gps_grid_rep[i][:length_g]
                grid_cls_token = grid_traj_rep[i].unsqueeze(0)
                gps_cls_grid_token = gps_traj_grid_rep[i].unsqueeze(0)

                # position
                position_r = torch.arange(length_r+1).long().cuda()
                pos_emb_r = self.position_embedding2_share(position_r)

                position_g = torch.arange(length_g+1).long().cuda()
                pos_emb_g = self.position_embedding2_share(position_g)

                # update route_emb
                route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
                modal_emb = modal_emb0.unsqueeze(0).repeat(length_r+1, 1)
                route_emb = route_emb + pos_emb_r + modal_emb
                route_emb = self.fc2_share(route_emb)

                # update grid_emb
                grid_emb = torch.cat([grid_cls_token, grid_road_token], dim=0)
                modal_emb = modal_emb2.unsqueeze(0).repeat(length_g+1, 1)
                grid_emb = grid_emb + pos_emb_g + modal_emb
                grid_emb = self.fc4_share(grid_emb)

                # update gps_emb_r
                gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
                modal_emb = modal_emb1.unsqueeze(0).repeat(length_r+1, 1)
                gps_emb = gps_emb + pos_emb_r + modal_emb
                gps_emb = self.fc2_share(gps_emb)

                # update gps_emb_g
                gps_g_emb = torch.cat([gps_cls_grid_token, gps_grid_token], dim=0)
                modal_emb = modal_emb3.unsqueeze(0).repeat(length_g+1, 1)
                gps_g_emb = gps_g_emb + pos_emb_g + modal_emb
                gps_g_emb = self.fc4_share(gps_g_emb)
                
                data = torch.cat([gps_emb, route_emb, gps_g_emb, grid_emb], dim=0)
                data_list.append(data)

                mask = torch.tensor([False] * data.shape[0]).cuda() # mask的位置为true
                mask_list.append(mask)
        else:
            data_list = []
            mask_list = []
            route_emb_list = []
            grid_emb_list = []
            gps_emb_list = []
            gps_g_emb_list = []

            mask_list_route_emb = []
            mask_list_grid_emb = []
            mask_list_gps_emb = []
            mask_list_gps_g_emb = []

            route_length = [length[length!=vocab_this].shape[0] for length in route_assign_mat]
            grid_length = [length[length!=grid_vocab_this].shape[0] for length in grid_assign_mat]

            modal_emb0 = self.modal_embedding_four_module(torch.tensor(0).cuda())
            modal_emb1 = self.modal_embedding_four_module(torch.tensor(1).cuda())
            modal_emb2 = self.modal_embedding_four_module(torch.tensor(2).cuda())
            modal_emb3 = self.modal_embedding_four_module(torch.tensor(3).cuda())

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):

                route_road_token = route_road_rep[i]
                gps_road_token = gps_road_rep[i]
                route_cls_token = route_traj_rep[i].unsqueeze(0)
                gps_cls_token = gps_traj_rep[i].unsqueeze(0)

                grid_road_token = grid_road_rep[i]
                gps_grid_token = gps_grid_rep[i]
                grid_cls_token = grid_traj_rep[i].unsqueeze(0)
                gps_cls_grid_token = gps_traj_grid_rep[i].unsqueeze(0)


                route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
                grid_emb = torch.cat([grid_cls_token, grid_road_token], dim=0)
                gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
                gps_g_emb = torch.cat([gps_cls_grid_token, gps_grid_token], dim=0)


                route_emb_list.append(route_emb)
                grid_emb_list.append(grid_emb)
                gps_emb_list.append(gps_emb)
                gps_g_emb_list.append(gps_g_emb)

                mask_route_emb = torch.tensor([False] * (length_r+1) + [True] * (route_emb.shape[0]-(length_r+1))).cuda()
                mask_list_route_emb.append(mask_route_emb)
                mask_grid_emb = torch.tensor([False] * (length_g+1) + [True] * (grid_emb.shape[0]-(length_g+1))).cuda()
                mask_list_grid_emb.append(mask_grid_emb)
                mask_gps_emb = torch.tensor([False] * (length_r+1) + [True] * (gps_emb.shape[0]-(length_r+1))).cuda()
                mask_list_gps_emb.append(mask_gps_emb)
                mask_gps_g_emb = torch.tensor([False] * (length_g+1) + [True] * (gps_g_emb.shape[0]-(length_g+1))).cuda()
                mask_list_gps_g_emb.append(mask_gps_g_emb)

            route_emb = torch.stack(route_emb_list, dim=0)
            mask_route_emb = torch.stack(mask_list_route_emb, dim=0)
            grid_emb = torch.stack(grid_emb_list, dim=0)
            mask_grid_emb = torch.stack(mask_list_grid_emb, dim=0)
            gps_emb = torch.stack(gps_emb_list, dim=0)
            mask_gps_emb = torch.stack(mask_list_gps_emb, dim=0)
            gps_g_emb = torch.stack(gps_g_emb_list, dim=0)
            mask_gps_g_emb = torch.stack(mask_list_gps_g_emb, dim=0)

            # 将输入张量的维度调整为 (S, N, E)，其中 S 是序列长度，N 是批次大小，E 是嵌入维度
            route_emb = route_emb.permute(1, 0, 2).cuda()
            gps_emb = gps_emb.permute(1, 0, 2).cuda()
            grid_emb = grid_emb.permute(1, 0, 2).cuda()
            gps_g_emb = gps_g_emb.permute(1, 0, 2).cuda()

            mask_route_emb = mask_route_emb.cuda()
            mask_grid_emb = mask_grid_emb.cuda()
            mask_gps_emb = mask_gps_emb.cuda()
            mask_gps_g_emb = mask_gps_g_emb.cuda()

            # Route embedding as query, others as key and value
            attn_route_to_gps = self.cross_attn_route_to_gps_share(route_emb, gps_emb, gps_emb,mask_route_emb, mask_gps_emb)
            attn_route_to_grid = self.cross_attn_route_to_grid_share(route_emb, grid_emb, grid_emb,mask_route_emb, mask_grid_emb)
            attn_route_to_gps_g = self.cross_attn_route_to_gps_g_share(route_emb, gps_g_emb, gps_g_emb,mask_route_emb, mask_gps_g_emb)
            # attn_route_to_route = self.cross_attn_route_to_route_share(route_emb, route_emb, route_emb,mask_route_emb, mask_route_emb)

            # GPS embedding as query, others as key and value
            attn_gps_to_route = self.cross_attn_gps_to_route_share(gps_emb, route_emb, route_emb,mask_gps_emb, mask_route_emb)
            attn_gps_to_grid = self.cross_attn_gps_to_grid_share(gps_emb, grid_emb, grid_emb,mask_gps_emb, mask_grid_emb)
            attn_gps_to_gps_g = self.cross_attn_gps_to_gps_g_share(gps_emb, gps_g_emb, gps_g_emb,mask_gps_emb, mask_gps_g_emb)
            # attn_gps_to_gps = self.cross_attn_gps_to_gps_share(gps_emb, gps_emb, gps_emb,mask_gps_emb, mask_gps_emb)

            # Grid embedding as query, others as key and value
            attn_grid_to_route = self.cross_attn_grid_to_route_share(grid_emb, route_emb, route_emb,mask_grid_emb, mask_route_emb)
            attn_grid_to_gps =self. cross_attn_grid_to_gps_share(grid_emb, gps_emb, gps_emb,mask_grid_emb, mask_gps_emb)
            attn_grid_to_gps_g = self.cross_attn_grid_to_gps_g_share(grid_emb, gps_g_emb, gps_g_emb,mask_grid_emb, mask_gps_g_emb)
            # attn_grid_to_grid = self.cross_attn_grid_to_grid_share(grid_emb, grid_emb, grid_emb,mask_grid_emb, mask_grid_emb)

            # GPS Grid embedding as query, others as key and value
            attn_gps_g_to_route = self.cross_attn_gps_g_to_route_share(gps_g_emb, route_emb, route_emb,mask_gps_g_emb, mask_route_emb)
            attn_gps_g_to_gps = self.cross_attn_gps_g_to_gps_share(gps_g_emb, gps_emb, gps_emb,mask_gps_g_emb, mask_gps_emb)
            attn_gps_g_to_grid = self.cross_attn_gps_g_to_grid_share(gps_g_emb, grid_emb, grid_emb,mask_gps_g_emb, mask_grid_emb)
            # attn_gps_g_to_gps_g = self.cross_attn_gps_g_to_gps_g_share(gps_g_emb, gps_g_emb, gps_g_emb,mask_gps_g_emb, mask_gps_g_emb)

            final_route_emb = attn_route_to_gps + attn_route_to_grid + attn_route_to_gps_g #+ attn_route_to_route
            final_gps_emb = attn_gps_to_route + attn_gps_to_grid + attn_gps_to_gps_g #+ attn_gps_to_gps
            final_grid_emb = attn_grid_to_route + attn_grid_to_gps + attn_grid_to_gps_g #+ attn_grid_to_grid
            final_gps_g_emb = attn_gps_g_to_route + attn_gps_g_to_gps + attn_gps_g_to_grid #+ attn_gps_g_to_gps_g

            final_route_emb = final_route_emb.permute(1, 0, 2)
            final_gps_emb = final_gps_emb.permute(1, 0, 2)
            final_grid_emb = final_grid_emb.permute(1, 0, 2)
            final_gps_g_emb = final_gps_g_emb.permute(1, 0, 2)

            non_route_padding_mask = ~mask_route_emb
            non_grid_padding_mask = ~mask_grid_emb
            non_gps_padding_mask = ~mask_gps_emb
            non_gps_g_padding_mask = ~mask_gps_g_emb

            # 对于每个 batch，使用非 padding mask 从 final_route_emb 中选择有效位置
            non_route_padding_emb = []
            non_grid_padding_emb = []
            non_gps_padding_emb = []
            non_gps_g_padding_emb = []

            for i, (length_r, length_g) in enumerate(zip(route_length, grid_length)):
                # 选择每个 batch 的非 padding 部分
                route_emb = final_route_emb[i][non_route_padding_mask[i]]
                non_route_padding_emb.append(route_emb)
                grid_emb = final_grid_emb[i][non_grid_padding_mask[i]]
                non_grid_padding_emb.append(grid_emb)
                gps_emb = final_gps_emb[i][non_gps_padding_mask[i]]
                non_gps_padding_emb.append(gps_emb)
                gps_g_emb = final_gps_g_emb[i][non_gps_g_padding_mask[i]]
                non_gps_g_padding_emb.append(gps_g_emb)


                # position
                position_r = torch.arange(length_r+1).long().cuda()
                pos_emb_r = self.position_embedding2_share(position_r) 

                position_g = torch.arange(length_g+1).long().cuda()
                pos_emb_g = self.position_embedding2_share(position_g) 

                # update route_emb
                modal_emb = modal_emb0.unsqueeze(0).repeat(length_r+1, 1)
                route_emb = route_emb + pos_emb_r + modal_emb
                route_emb = self.fc2_share(route_emb)

                # update grid_emb
                modal_emb = modal_emb2.unsqueeze(0).repeat(length_g+1, 1)
                grid_emb = grid_emb + pos_emb_g + modal_emb
                grid_emb = self.fc4_share(grid_emb)

                # update gps_emb_r
                modal_emb = modal_emb1.unsqueeze(0).repeat(length_r+1, 1)
                gps_emb = gps_emb + pos_emb_r + modal_emb
                gps_emb = self.fc2_share(gps_emb)

                # update gps_emb_g
                modal_emb = modal_emb3.unsqueeze(0).repeat(length_g+1, 1)
                gps_g_emb = gps_g_emb + pos_emb_g + modal_emb
                gps_g_emb = self.fc4_share(gps_g_emb)

                data = torch.cat([gps_emb, route_emb, gps_g_emb, grid_emb], dim=0)
                data_list.append(data)
                mask = torch.tensor([False] * data.shape[0]).cuda() # mask的位置为true
                mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        joint_emb = self.sharedtransformer_share(joint_data, None, mask_mat)
        joint_emb = joint_emb.permute(1, 0, 2)
        # 每一行的0 和 length+1 对应的是 gps_traj_rep 和 route_traj_rep
        # 2length+1 和 2length+length_g+1 对应的是 gps_traj_grid_rep 和 grid_traj_rep
        # 因为有cls
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack([joint_emb[i, length+1] for i, length in enumerate(route_length)], dim=0)
        gps_traj_grid_rep = torch.stack([joint_emb[i, 2*length+2] for i, length in enumerate(route_length)], dim=0)
        grid_traj_rep = torch.stack([joint_emb[i, 2*length_r+length_g+3] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))], dim=0)

        gps_road_rep = rnn_utils.pad_sequence([joint_emb[i, 1:length+1] for i, length in enumerate(route_length)],
                                              padding_value=0, batch_first=True)
        route_road_rep = rnn_utils.pad_sequence([joint_emb[i, length+2:2*length+2] for i, length in enumerate(route_length)],
                                                padding_value=0, batch_first=True)
        gps_grid_rep = rnn_utils.pad_sequence([joint_emb[i, 2*length_r+3:2*length_r+length_g+3] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))],
                                                padding_value=0, batch_first=True)
        grid_road_rep = rnn_utils.pad_sequence([joint_emb[i, 2*length_r+length_g+4:2*length_r+2*length_g+4] for i, (length_r, length_g) in enumerate(zip(route_length, grid_length))],
                                                padding_value=0, batch_first=True)

        return gps_traj_rep, route_traj_rep, gps_traj_grid_rep, grid_traj_rep, gps_road_rep, route_road_rep, gps_grid_rep, grid_road_rep
    
    def forward(self, route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, \
                grid_data, masked_grid_assign_mat, gps_data_grid, masked_gps_assign_mat_grid, grid_assign_mat, gps_length_grid, source_city):
        
        if source_city:
            gps_road_rep, gps_traj_rep = self.encode_gps(gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length)
            gps_grid_rep, gps_traj_grid_rep = self.encode_gps_grid(gps_data_grid, masked_gps_assign_mat_grid, masked_grid_assign_mat, gps_length_grid)

            route_road_rep, route_traj_rep = self.encode_route(route_data, route_assign_mat, masked_route_assign_mat)
            grid_road_rep, grid_traj_rep, prior_embedding = self.encode_grid(grid_data, grid_assign_mat, masked_grid_assign_mat)


            if self.add_pgmnet_city:
                xian_tensor = torch.tensor([
                    0.5112, 0.5372, 0.5000, 0.5161, 0.5135, 0.6629, 0.6203, 0.6750, 0.6731,
                    0.6765, 0.4972, 0.4368, 0.5207, 0.5176, 0.5079, 0.5230, 0.5116, 0.5278,
                    0.5270, 0.5234, 0.4773, 0.4349, 0.4976, 0.4902, 0.4791
                ], dtype=torch.float32).repeat(prior_embedding.shape[0], 1).cuda()
                domain = self.domain_linear(xian_tensor)
                gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep = \
                    self.pgmnet_city(domain, gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep)
            if self.add_pgmnet_traj:
                prior = self.prior_linear(prior_embedding)
                gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep = \
                    self.pgmnet_city(prior, gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep)

            gps_traj_joint_rep, route_traj_joint_rep, gps_traj_grid_joint_rep, grid_traj_joint_rep, gps_road_joint_rep, route_road_joint_rep, \
                gps_grid_joint_rep, grid_road_joint_rep = self.encode_joint_four_stream(route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, \
                                    grid_road_rep, grid_traj_rep, gps_grid_rep, gps_traj_grid_rep, grid_assign_mat)
        else:
            gps_road_rep, gps_traj_rep = self.encode_gps_extra(gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length)
            gps_grid_rep, gps_traj_grid_rep = self.encode_gps_grid_extra(gps_data_grid, masked_gps_assign_mat_grid, masked_grid_assign_mat, gps_length_grid)

            route_road_rep, route_traj_rep = self.encode_route_extra(route_data, route_assign_mat, masked_route_assign_mat)
            grid_road_rep, grid_traj_rep, prior_embedding = self.encode_grid_extra(grid_data, grid_assign_mat, masked_grid_assign_mat)

            if self.add_pgmnet_city:
                chengdu_tensor = torch.tensor([
                    0.4888, 0.4628, 0.5000, 0.4839, 0.4865, 0.3371, 0.3797, 0.3250, 0.3269,
                    0.3235, 0.5028, 0.5632, 0.4793, 0.4824, 0.4921, 0.4770, 0.4884, 0.4722,
                    0.4730, 0.4766, 0.5227, 0.5651, 0.5024, 0.5098, 0.5209
                ], dtype=torch.float32).repeat(prior_embedding.shape[0], 1).cuda()
                domain = self.domain_linear(chengdu_tensor)
                gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep = \
                    self.pgmnet_city(domain, gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep)
            if self.add_pgmnet_traj:
                prior = self.prior_linear(prior_embedding)
                gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep = \
                    self.pgmnet_city(prior, gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep)

            gps_traj_joint_rep, route_traj_joint_rep, gps_traj_grid_joint_rep, grid_traj_joint_rep, gps_road_joint_rep, route_road_joint_rep, \
                gps_grid_joint_rep, grid_road_joint_rep = self.encode_joint_four_stream_extra(route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, \
                                    grid_road_rep, grid_traj_rep, gps_grid_rep, gps_traj_grid_rep, grid_assign_mat)
            
        gps_traj_joint_rep_share, route_traj_joint_rep_share, gps_traj_grid_joint_rep_share, grid_traj_joint_rep_share, gps_road_joint_rep_share, route_road_joint_rep_share, \
                gps_grid_joint_rep_share, grid_road_joint_rep_share = self.city_share_encode_joint(route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, \
                                    grid_road_rep, grid_traj_rep, gps_grid_rep, gps_traj_grid_rep, grid_assign_mat, source_city)    
        if self.add_pgmnet_traj_2:
            gps_traj_joint_rep, gps_traj_grid_joint_rep, route_traj_joint_rep, grid_traj_joint_rep, gps_road_joint_rep, \
                                gps_grid_joint_rep, route_road_joint_rep, grid_road_joint_rep = \
                self.pgmnet_traj_2(prior, gps_traj_joint_rep, gps_traj_grid_joint_rep, route_traj_joint_rep, grid_traj_joint_rep, gps_road_joint_rep, \
                                gps_grid_joint_rep, route_road_joint_rep, grid_road_joint_rep)
            gps_traj_joint_rep_share, gps_traj_grid_joint_rep_share, route_traj_joint_rep_share, grid_traj_joint_rep_share, gps_road_joint_rep_share, \
                                gps_grid_joint_rep_share, route_road_joint_rep_share, grid_road_joint_rep_share = \
                self.pgmnet_traj_2(prior, gps_traj_joint_rep_share, gps_traj_grid_joint_rep_share, route_traj_joint_rep_share, grid_traj_joint_rep_share, gps_road_joint_rep_share, \
                                gps_grid_joint_rep_share, route_road_joint_rep_share, grid_road_joint_rep_share)



        return gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, gps_grid_rep, gps_traj_grid_rep, grid_road_rep, grid_traj_rep, \
               gps_traj_joint_rep, route_traj_joint_rep, gps_traj_grid_joint_rep, grid_traj_joint_rep, gps_road_joint_rep, route_road_joint_rep, \
              gps_grid_joint_rep, grid_road_joint_rep, gps_traj_joint_rep_share, route_traj_joint_rep_share, gps_traj_grid_joint_rep_share, grid_traj_joint_rep_share, gps_road_joint_rep_share, route_road_joint_rep_share, \
                gps_grid_joint_rep_share, grid_road_joint_rep_share

# GAT
class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        # update road edge features using GAT
        self.layer1 = GATConv(input_size, output_size)
        self.layer2 = GATConv(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x

class TransformerModel(nn.Module):  # vanilla transformer
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = src.permute(1, 0, 2)  # 从 (32, 47, 256) 调整为 (47, 32, 256)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output

# Continuous time embedding
class IntervalEmbedding(nn.Module):
    def __init__(self, num_bins, hidden_size):
        super(IntervalEmbedding, self).__init__()
        self.layer1 = nn.Linear(1, num_bins)
        self.emb = nn.Embedding(num_bins, hidden_size)
        self.activation = nn.Softmax()
    def forward(self, x):
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output


class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, query, key, value, mask_q=None, mask_k=None):
        batch_size = query.size(1)  # 批处理大小
        seq_q, seq_k = query.size(0), key.size(0)  # 序列长度


        # Create attention mask
        if mask_q is not None and mask_k is not None:
            attn_mask = torch.full((batch_size, seq_q, seq_k), float('-inf'))  # Initialize with -inf

            for b in range(batch_size):
                # Set valid positions to 0
                attn_mask[b, :, ~mask_k[b]] = 0  # mask_k 是布尔掩码，~mask_k 表示相反的掩码
                attn_mask[b, ~mask_q[b], :] = 0  # mask_q 是布尔掩码，~mask_q 表示相反的掩码

            attn_mask = attn_mask.to(query.device)  # Move to the same device as query

            # Expand attn_mask for multi-head attention
            num_heads = self.attention.num_heads
            attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(batch_size * num_heads, seq_q, seq_k)
        else:
            attn_mask = None

        # Compute attention
        attention_output, _ = self.attention(query, key, value, attn_mask=attn_mask)
        return attention_output

import torch.nn.functional as F

class GateNU(nn.Module):
    def __init__(self, hidden_units, gamma=2.0, l2_reg=0.0):
        super(GateNU, self).__init__()
        
        assert len(hidden_units) == 2
        
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.hidden_units = hidden_units

        # Define the layers
        self.dense_layers = nn.ModuleList([
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.Linear(hidden_units[1], hidden_units[1])
        ])
    
    def forward(self, x):
        # Apply the first dense layer with ReLU
        x = F.relu(self.dense_layers[0](x))
        # Apply the second dense layer with Sigmoid and gamma scaling
        x = self.gamma * torch.sigmoid(self.dense_layers[1](x))
        
        return x
    
class PGMNet(nn.Module):
    def __init__(self, domain_shape, emb_shape, l2_reg=0.0):
        super(PGMNet, self).__init__()
        
        self.l2_reg = l2_reg
        self.gate_nu = GateNU(hidden_units=[domain_shape + emb_shape, emb_shape], gamma=2.0, l2_reg=self.l2_reg)
    
    def forward(self, domain, gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep,
                gps_grid_rep, route_road_rep, grid_road_rep):
            emb = torch.cat([gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep], dim=1)
            x = torch.cat([domain, emb.detach()], dim=-1)
            delta = self.gate_nu(x)
            
            delta_0, delta_1, delta_2, delta_3 = delta.split(256, dim=1)
            gps_traj_rep = gps_traj_rep * delta_0
            gps_traj_grid_rep = gps_traj_grid_rep * delta_1
            route_traj_rep = route_traj_rep * delta_2
            grid_traj_rep = grid_traj_rep * delta_3
            
            gps_road_rep = gps_road_rep * delta_0.unsqueeze(-2).expand(-1, gps_road_rep.shape[-2], -1)
            gps_grid_rep = gps_grid_rep * delta_1.unsqueeze(-2).expand(-1, gps_grid_rep.shape[-2], -1)
            route_road_rep = route_road_rep * delta_2.unsqueeze(-2).expand(-1, route_road_rep.shape[-2], -1)
            grid_road_rep = grid_road_rep * delta_3.unsqueeze(-2).expand(-1, grid_road_rep.shape[-2], -1)
            
            return gps_traj_rep, gps_traj_grid_rep, route_traj_rep, grid_traj_rep, gps_road_rep, gps_grid_rep, route_road_rep, grid_road_rep

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),   
            nn.ReLU(),
            nn.Linear(128, output_dim)   
        )
        
    def forward(self, x):
        return self.fc(x)