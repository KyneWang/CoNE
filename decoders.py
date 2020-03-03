import torch
import torch.nn as nn
import math

def load_decoder(name, config):
    if name == "TransE":
        return TransE(config)
    elif name == "ConvE":
        return ConvE(config)
    elif name == "RotatE":
        return RotatE(config)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

class TransE(Decoder):
    def __init__(self, config):
        super(TransE, self).__init__(config)

    def forward(self, head, relation, tail, mode):
        if mode == 'head_batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = torch.norm(score, p=1, dim=2)
        return score

    def loss(self, d_good, d_bad):
        loss = nn.functional.relu(self.config.margin + d_good - d_bad)
        return loss


class ConvE(Decoder):
    def __init__(self, config):
        super(ConvE, self).__init__(config)
        self.num_entities = config.entityNum
        self.num_relations = config.relationNum
        self.window_size = 3
        self.out_channel = 32
        self.stride = 1
        self.padding = 0
        self.image_shape = math.sqrt(self.config.embedding_dim * 2)
        self.output_size = int((self.image_shape - self.window_size + 2 * self.padding) / self.stride + 1)
        self.fc_hiddensize = self.output_size * self.output_size * self.out_channel

        self.inp_drop = torch.nn.Dropout(config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, self.out_channel, (self.window_size, self.window_size), self.stride,
                                     self.padding, bias=config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bnt = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channel)
        self.bn2 = torch.nn.BatchNorm1d(config.embedding_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.config.nagTriple_num + 1)))
        self.fc = torch.nn.Linear(self.fc_hiddensize, config.embedding_dim)  # 10368
        self.fc2 = torch.nn.Linear(21888, config.embedding_dim)  # for nei ??

    def forward(self,  head, relation, tail, mode):
        if mode == 'head_batch':
            e1_embedded = tail.view(-1, 1, 10, 20)  # [N, 1, 10, 20]
            rel_embedded = relation.view(-1, 1, 10, 20)  # [N, 1, 10, 20]
            ans_embeded = head
        else:
            e1_embedded = head.view(-1, 1, 10, 20)  # [N, 1, 10, 20]
            rel_embedded = relation.view(-1, 1, 10, 20)  # [N, 1, 10, 20]
            ans_embeded = tail
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)  # [N, 1, 20, 20]
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)  # [N, 32, 18, 18]
        x = nn.functional.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(len(relation), -1)  # [N, 10368]
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)  # [N, D]
        x = torch.bmm(x.unsqueeze(1), ans_embeded.permute(0,2,1)).squeeze(1)  # [N, E]
        x += self.b.expand_as(x)  # [N, E]
        score = torch.sigmoid(x)
        return score

    def loss(self, d_good, d_bad):
        label = torch.cat([torch.ones_like(d_good), torch.zeros_like(d_bad)], dim=1).float()
        score = torch.cat([d_good, d_bad], dim=1)
        loss = - 1 * torch.mean(label * torch.log(score + 1e-30) + (1 - label) * torch.log(1 - score + 1e-30))
        return loss

class RotatE(Decoder):
    def __init__(self, config):
        super(RotatE, self).__init__(config)
        self.epsilon = 2.0
        self.embedding_range = (self.config.margin + self.epsilon) / config.embedding_dim

    def forward(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        relation, relation2 = torch.chunk(relation, 2, dim=2)

        phase_relation = relation / (self.embedding_range / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head_batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.config.margin - score.sum(dim=2)
        return score

    def loss(self, d_good, d_bad):
        positive_score = nn.functional.logsigmoid(d_good).squeeze(dim=1)
        negative_score = (nn.functional.softmax(d_bad, dim=1).detach()
                          * nn.functional.logsigmoid(-d_bad)).sum(dim=1)
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss
