import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, uniform_
from encoders import load_encoder
from decoders import load_decoder

class preTrain_model(nn.Module):
    def __init__(self, config):
        super(preTrain_model, self).__init__()
        self.config = config
        self.decoder_name = config.decoder_name
        self.rel_embed = nn.Embedding(config.relationNum, config.embedding_dim)
        self.ent_embed = nn.Embedding(config.entityNum, config.embedding_dim)
        self.decoder = load_decoder(self.decoder_name, config)

    def init(self, eMat=None, rMat = None):
        if type(eMat) != type(None):
            self.ent_embed.weight.data.copy_(torch.from_numpy(eMat[:,:self.config.embedding_dim]))
        else:
            if self.decoder_name == "RotatE":
                uniform_(self.ent_embed.weight.data, a=-1 * self.embedding_range, b=self.embedding_range)
            else:
                self.ent_embed.weight.data.normal_(1 / self.ent_embed.weight.size(1) ** 0.5)
                self.ent_embed.weight.data.renorm_(2, 0, 1)

        if type(rMat) != type(None):
            self.rel_embed.weight.data.copy_(torch.from_numpy(rMat[:,:self.config.embedding_dim]))
        else:
            if self.decoder_name == "RotatE":
                uniform_(self.rel_embed.weight.data, a=-1 * self.embedding_range, b=self.embedding_range)
            else:
                self.rel_embed.weight.data.normal_(1 / self.rel_embed.weight.size(1) ** 0.5)
                self.rel_embed.weight.data.renorm_(2, 0, 1)

    def constraint(self):
        if self.decoder_name == "TransE" or self.decoder_name == "RotatE":
            self.ent_embed.weight.data.renorm_(2, 0, 1)
            self.rel_embed.weight.data.renorm_(2, 0, 1)

    def forward(self, src, rel, dst, mode):
        src_embeded = self.ent_embed(src)
        dst_embeded = self.ent_embed(dst)
        rel_embeded = self.rel_embed(rel)
        output = self.decoder(src_embeded, rel_embeded, dst_embeded, mode)
        return output

    def loss(self, src, dst, rel, src_bad, dst_bad, mode):
        d_good = self.forward(src, rel, dst, mode)
        d_bad = self.forward(src_bad, rel, dst_bad, mode)
        return self.decoder.loss(d_good, d_bad)

    def score(self, src, rel, dst, mode):
        return self.forward(src, rel, dst, mode)

class CoNE_model(nn.Module):
    def __init__(self, config):
        super(CoNE_model, self).__init__()
        self.config = config
        self.encoder_name = config.encoder_name
        self.decoder_name = config.decoder_name
        self.rel_embed = nn.Embedding(config.relationNum, config.embedding_dim)
        self.ent_embed = nn.Embedding(config.entityNum, config.embedding_dim)
        self.nei_embed = nn.Embedding(config.entityNum, config.embedding_dim)
        self.weight_embed = nn.Embedding(config.entityNum, 1)
        self.neiMatrix = nn.Embedding(config.entityNum, config.nei_size)

        self.encoder = load_encoder(self.encoder_name, config)
        self.decoder = load_decoder(self.decoder_name, config)


    def init(self, neiMat = None, eMat=None, rMat = None):
        if type(eMat) != type(None):
            self.ent_embed.weight.data.copy_(torch.from_numpy(eMat[:,:self.config.embedding_dim]))
        else:
            if self.decoder_name == "RotatE":
                uniform_(self.ent_embed.weight.data, a=-1 * self.embedding_range, b=self.embedding_range)
            if self.decoder_name == "TransE" and self.config.dataset == "WN18":
                xavier_uniform_(self.ent_embed.weight.data, gain=1)
            else:
                self.ent_embed.weight.data.normal_(1 / self.ent_embed.weight.size(1) ** 0.5)
                self.ent_embed.weight.data.renorm_(2, 0, 1)

        if type(rMat) != type(None):
            self.rel_embed.weight.data.copy_(torch.from_numpy(rMat[:,:self.config.embedding_dim]))
        else:
            if self.decoder_name == "RotatE":
                uniform_(self.rel_embed.weight.data, a=-1 * self.embedding_range, b=self.embedding_range)

            if self.decoder_name == "TransE" and self.config.dataset == "WN18":
                xavier_uniform_(self.rel_embed.weight.data, gain=1)
            else:
                self.rel_embed.weight.data.normal_(1 / self.rel_embed.weight.size(1) ** 0.5)
                self.rel_embed.weight.data.renorm_(2, 0, 1)

        neiMatrix_tensor = torch.from_numpy(neiMat).long()
        self.neiMatrix.weight.data.copy_(neiMatrix_tensor)
        self.neiMatrix.weight.require_grad = False
        self.nei_embed.weight.data.normal_(1 / self.nei_embed.weight.size(1) ** 0.5)
        self.nei_embed.weight.data.renorm_(2, 0, 1)
        self.weight_embed.weight.data.normal_(1 / self.weight_embed.weight.size(1) ** 0.5)
        self.weight_embed.weight.data.renorm_(2, 0, 1)

    def constraint(self):
        if self.decoder_name == "TransE" or self.decoder_name == "RotatE":
            self.ent_embed.weight.data.renorm_(2, 0, 1)
            self.rel_embed.weight.data.renorm_(2, 0, 1)

    def forward(self, src, rel, dst, mode):
        if mode == "head_batch":
            input_ent, predict_ent  = dst, src
        else:
            input_ent, predict_ent = src, dst

        input_ent = input_ent.view(-1)
        input_nei = self.neiMatrix(input_ent).long()
        nei_mask = (input_nei > 0).float()
        input_ent_embeded = self.ent_embed(input_ent)
        predict_ent_embeded = self.ent_embed(predict_ent)
        nei_embeded_key = self.nei_embed(input_nei)
        nei_embeded_value = self.nei_embed(input_nei)
        rel_embeded = self.rel_embed(rel.view(-1))

        nei_encode_embeded = self.encoder(input_ent_embeded, rel_embeded, nei_embeded_key, nei_embeded_value, nei_mask, dim=2)
        nei_encode_embeded = nei_encode_embeded.view(src.size()[0], -1, self.config.embedding_dim)
        input_ent_embeded = input_ent_embeded.unsqueeze(1)
        rel_embeded = rel_embeded.unsqueeze(1)
        #input_ent_embeded = (input_ent_embeded + nei_encode_embeded) / 2
        weight = torch.nn.functional.sigmoid(self.weight_embed(input_ent)).unsqueeze(2)
        input_ent_embeded = weight * nei_encode_embeded + (1 - weight) * input_ent_embeded

        if mode == "head_batch":
            output = self.decoder(predict_ent_embeded, rel_embeded, input_ent_embeded, mode)
        else:
            output = self.decoder(input_ent_embeded, rel_embeded, predict_ent_embeded, mode)
        return output

    def loss(self, src, dst, rel, src_bad, dst_bad, mode):
        d_good = self.forward(src, rel, dst, mode)
        d_bad = self.forward(src_bad, rel, dst_bad, mode)
        return self.decoder.loss(d_good, d_bad)

    def score(self, src, rel, dst, mode):
        return self.forward(src, rel, dst, mode)