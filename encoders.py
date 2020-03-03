import torch
import torch.nn as nn

def load_encoder(name, config):
    if name == "CBOW":
        return CBOW_Encoder(config)
    elif name == "GCN":
        return GCN_Encoder(config)
    elif name == "DMN":
        return DMN_Encoder(config)
    elif name == "GMN":
        return GMN_Encoder(config)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

class CBOW_Encoder(Encoder):
    def __init__(self, config):
        super(CBOW_Encoder, self).__init__(config)

    def forward(self, e1_embeded, rel_embeded, nei_embeded_key, nei_embeded_value, nei_mask, dim):
        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        item_length = nei_mask.sum(-1).unsqueeze(-1)
        feature_masked = nei_embeded_value * nei_mask.unsqueeze(-1)  # [N, 20, 200]
        sumVectors = torch.sum(feature_masked, dim-1)
        item_length = torch.max(item_length, torch.ones_like(item_length))
        output = (sumVectors / item_length)
        return output

class GCN_Encoder(Encoder):
    def __init__(self, config):
        super(GCN_Encoder, self).__init__(config)
        in_features, out_features = self.config.embedding_dim, self.config.embedding_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, e1_embeded, rel_embeded, nei_embeded_key, nei_embeded_value, nei_mask, dim):
        batch_size = nei_embeded_value.size()[0]
        feature_masked = nei_embeded_value * nei_mask.unsqueeze(-1)  # [N, 20, 200]
        feature_viewed = feature_masked.view(-1, self.in_features)
        output = torch.nn.functional.tanh(torch.mm(feature_viewed, self.weight))
        output = output.view([batch_size, -1, self.in_features])
        sumVectors = torch.sum(output, dim-1)
        return sumVectors

class DMN_Encoder(Encoder):
    def __init__(self, config):
        super(DMN_Encoder, self).__init__(config)
        embedding_dim = self.config.embedding_dim
        max_hops = 3
        self.max_hops = max_hops
        self.embedding_dim = embedding_dim
        self.linfc = torch.nn.Linear(embedding_dim, embedding_dim)
        self.attfc = torch.nn.Linear(embedding_dim * 2, 1)

    def l2_norm(self, x, dim = 1):
        result = x.renorm(p=2, dim=dim-1, maxnorm=1e-12).mul(1/1e-12)
        return result

    def _masked_softmax(self, value_batch, mask_batch, dim = 1):
        max_v_batch = torch.max(value_batch, dim = dim)[0].unsqueeze(dim) # avoid Nan
        exp_um_batch = torch.exp(value_batch - max_v_batch)  # [N, n]
        mask_batch = mask_batch.float()
        masked_batch = exp_um_batch * mask_batch # [N, n]
        sum_2d_batch = torch.sum(masked_batch, dim = dim).unsqueeze(dim) + 1e-5   # [N, 1]
        return masked_batch / sum_2d_batch  # [N, n]  division

    def forward(self, e1_embeded, rel_embeded, nei_embeded_key, nei_embeded_value, nei_mask, dim):
        batch_size = nei_embeded_value.shape[0]
        features = nei_embeded_value * nei_mask.unsqueeze(-1)  # [N, 20, 200]
        u = list()
        u.append(e1_embeded)  # [N, 200]
        for hop in range(self.max_hops):
            u_batch =  torch.nn.functional.relu(self.linfc(u[-1]))
            m_batch = torch.cat([features, u[-1].unsqueeze(1).expand_as(features)], dim)
            m_batch = m_batch.view(-1, self.embedding_dim * 2)
            att_batch = torch.nn.functional.relu(self.attfc(m_batch))
            att_batch = att_batch.view(batch_size, -1)
            p_batch = self._masked_softmax(att_batch, nei_mask, dim-1)
            o_batch = (features * p_batch.unsqueeze(-1)).sum(dim-1)  # [N, d]
            u.append(u_batch + o_batch)
        output = u[-1]
        return output

class GMN_Encoder(Encoder):
    def __init__(self, config):
        super(GMN_Encoder, self).__init__(config)
        embedding_dim = self.config.embedding_dim
        max_hops = 3
        self.max_hops = max_hops
        self.embedding_dim = embedding_dim
        self.Wk = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        torch.nn.init.xavier_uniform_(self.Wk, gain=1)
        self.Wv = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        torch.nn.init.xavier_uniform_(self.Wv, gain=1)

        self.linfc = torch.nn.Linear(embedding_dim, embedding_dim)
        self.attfc = torch.nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = torch.nn.Dropout(0.8)

    def _masked_softmax(self, value_batch, mask_batch, dim = 1):
        max_v_batch = torch.max(value_batch, dim = dim)[0].unsqueeze(dim) # avoid Nan
        exp_um_batch = torch.exp(value_batch - max_v_batch)  # [N, n]
        mask_batch = mask_batch.float()
        masked_batch = exp_um_batch * mask_batch # [N, n]
        sum_2d_batch = torch.sum(masked_batch, dim = dim).unsqueeze(dim) + 1e-5   # [N, 1]
        return masked_batch / sum_2d_batch  # [N, n]  division

    def l2_norm(self, x, dim = 1):
        result = x.renorm(p=2, dim=dim-1, maxnorm=1e-12).mul(1/1e-12)
        return result

    def forward(self, e1_embeded, rel_embeded, nei_embeded_key, nei_embeded_value, nei_mask, dim):
        batch_size = nei_embeded_key.shape[0]
        features_key = nei_embeded_key * nei_mask.unsqueeze(-1)  # [N, 20, 200]
        features_key = torch.matmul(features_key, self.Wk)
        features_value = nei_embeded_value * nei_mask.unsqueeze(-1)  # [N, 20, 200]
        features_value = torch.matmul(features_value, self.Wv)
        rel_batch = rel_embeded.unsqueeze(1).expand_as(features_key)
        u = list()
        u.append(e1_embeded)  # [N, 200]
        for hop in range(self.max_hops):
            u_batch = torch.nn.functional.leaky_relu(self.linfc(u[-1]))
            ent_batch = u[-1].unsqueeze(1).expand_as(features_key)
            m_batch = torch.cat([features_key, ent_batch, rel_batch], dim)
            m_batch = m_batch.view(-1, self.embedding_dim * 3)
            att_batch = torch.sum(torch.nn.functional.leaky_relu(self.attfc(m_batch)), dim=1)
            att_batch = att_batch.view(batch_size, -1)
            p_batch = self._masked_softmax(att_batch, nei_mask, dim-1)  #* 100
            o_batch = (features_value * p_batch.unsqueeze(-1)).sum(dim-1)  # [N, d]
            o_next = self.l2_norm(u_batch + o_batch)
            u.append(o_next)
        output = u[-1]
        return output
