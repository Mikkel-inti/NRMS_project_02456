import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    """Attention Layer (similar to AttLayer2 and SelfAttention in TensorFlow)."""

    def __init__(self, hidden_dim, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.fc(x), dim=1)
        return torch.sum(x * attn_weights, dim=1)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention Layer."""

    def __init__(self, head_num, head_dim, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.head_num = head_num
        self.head_dim = head_dim
        self.query = nn.Linear(head_dim, head_dim * head_num)
        self.key = nn.Linear(head_dim, head_dim * head_num)
        self.value = nn.Linear(head_dim, head_dim * head_num)
        self.fc = nn.Linear(head_num * head_dim, head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.query(x).view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).view(batch_size, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)
        return self.fc(context)


class NewsEncoder(nn.Module):
    """News Encoder."""

    def __init__(self, embedding_dim, vocab_size, pretrained_embedding=None, dropout=0.2, head_num=4, head_dim=64, attn_hidden_dim=128, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        if pretrained_embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embedding), freeze=False)

        self.dropout = nn.Dropout(dropout)
        self.self_attention = MultiHeadSelfAttention(head_num, head_dim, seed)
        self.attention_layer = AttentionLayer(attn_hidden_dim, seed)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = self.attention_layer(x)
        return x


class UserEncoder(nn.Module):
    """User Encoder."""

    def __init__(self, news_encoder, history_size, title_size, head_num=4, head_dim=64, attn_hidden_dim=128, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.news_encoder = news_encoder
        self.self_attention = MultiHeadSelfAttention(head_num, head_dim, seed)
        self.attention_layer = AttentionLayer(attn_hidden_dim, seed)

    def forward(self, history_titles):
        batch_size, history_size, title_size = history_titles.size()
        news_repr = torch.stack([self.news_encoder(history_titles[:, i]) for i in range(history_size)], dim=1)
        news_repr = self.self_attention(news_repr)
        user_repr = self.attention_layer(news_repr)
        return user_repr


class NRMSModel(nn.Module):
    """NRMS Model."""

    def __init__(self, hparams, word2vec_embedding=None, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.news_encoder = NewsEncoder(
            embedding_dim=hparams["word_emb_dim"],
            vocab_size=hparams["vocab_size"],
            pretrained_embedding=word2vec_embedding,
            dropout=hparams["dropout"],
            head_num=hparams["head_num"],
            head_dim=hparams["head_dim"],
            attn_hidden_dim=hparams["attention_hidden_dim"],
            seed=seed,
        )
        self.user_encoder = UserEncoder(
            news_encoder=self.news_encoder,
            history_size=hparams["history_size"],
            title_size=hparams["title_size"],
            head_num=hparams["head_num"],
            head_dim=hparams["head_dim"],
            attn_hidden_dim=hparams["attention_hidden_dim"],
            seed=seed,
        )

    def forward(self, history_titles, candidate_titles):
        user_repr = self.user_encoder(history_titles)
        candidate_repr = torch.stack([self.news_encoder(candidate_titles[:, i]) for i in range(candidate_titles.size(1))], dim=1)
        scores = torch.bmm(candidate_repr, user_repr.unsqueeze(-1)).squeeze(-1)
        return scores
