import argparse
import numpy as np

from models.CNN_model import *



class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        self.num_features = 512
        if self.args.dataset in ['usc-had']:
            # self.encoder = resnet18_gait()
            self.encoder = CNNNet18(in_planes=6, out_planes=128)
            self.num_features = 128
        elif self.args.dataset in ['our-ag']:
            self.encoder = CNNNet18(in_planes=6, out_planes=256)
            self.num_features = 256
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc1 = nn.Linear(self.num_features, self.args.num_actions, bias=False)
        self.fc2 = nn.Linear(self.num_features, self.args.num_actions * self.args.num_classes, bias=False)
        hdim = self.num_features
        self.attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.self_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)


    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc2.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc2(x)

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            if self.args.project == 'base':
                input = self.forward_metric(input)
                return input
            # else:
            #     print("meta mode")
            #     if self.args.dataset in ['uci-har', 'usc-had']:
            #         subject_idx, action_idx, query_idx = input
            #         logits = self._forward_t(subject_idx, action_idx, query_idx)
            #         return logits
            #     support_idx, query_idx = input
            #     logits = self._forward(support_idx, query_idx)
            #     return logits
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def _forward_t(self, proto_all, subject_index, action_index, query):
        proto_all, query_all, num_proto = self.get_proto(proto_all, query)

        logits0 = F.cosine_similarity(query_all, proto_all, dim=-1)
        logits0 = logits0 * self.args.temperature
        combined = torch.cat([proto_all, query_all], 1)  # Nk x (N + 1) x d, batch_size = NK

        subject_proto, num_sub_proto = self.get_proto_by_index(proto_all, subject_index)

        combined1 = self.self_attn(combined, subject_proto, subject_proto)
        proto1, query1 = combined1.split(num_proto, 1)
        logits1 = F.cosine_similarity(query1, proto1, dim=-1)
        logits1 = logits1 * self.args.temperature

        action_proto, num_act_proto = self.get_proto_by_index(proto_all, action_index)
        combined2 = self.attn(combined, action_proto, action_proto)
        proto2, query2 = combined2.split(num_proto, 1)
        logits2 = F.cosine_similarity(query2, proto2, dim=-1)
        logits2 = logits2 * self.args.temperature

        combined = combined1 + combined2
        proto, query = combined.split(num_proto, 1)
        logits3 = F.cosine_similarity(query, proto, dim=-1)
        logits3 = logits3 * self.args.temperature

        logits = logits0, logits1, logits2, logits3

        return logits

    def get_proto_by_index(self, proto_all, index):

        unique_label = torch.unique(index)
        proto = []
        for i in unique_label:
            ind = torch.nonzero(index == i).squeeze()
            embedding = proto_all[:, ind, :] if ind.numel() == 1 else proto_all[:, ind, :].mean(1)
            proto.append(embedding)
        proto = torch.stack(proto, dim=0)
        proto = proto.permute(1, 0, 2).contiguous().view(proto_all.shape[0], len(unique_label), -1)
        num_proto = proto.shape[1]
        return proto, num_proto

    def get_proto(self, proto, query):
        emb_dim = proto.size(-1)
        # get mean of the support
        proto = proto.mean(dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1] * query.shape[2]  # num of query*way

        query = query.view(-1, emb_dim).unsqueeze(1)

        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto = proto.view(num_batch * num_query, num_proto, emb_dim)
        return proto, query, num_proto

    def update_fc(self, dataloader, class_list):
        for batch in dataloader:
            data, label = [_.cuda() if torch.cuda.is_available() else _ for _ in batch]
            data = self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []

        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc2.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)

        return new_fc


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        _, max_ind = torch.max(attn, dim=1, keepdim=True)
        mask = torch.zeros_like(attn)
        mask.scatter_(1, max_ind, 1)
        attn = mask * attn
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

