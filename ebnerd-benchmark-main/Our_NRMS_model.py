import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAdditiveAttention(nn.Module):
    """Multi-Head Additive Attention Layer."""

    def __init__(self, head_num, head_dim, attn_hidden_dim=128, dropout=0.2, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.head_num = head_num
        self.head_dim = head_dim
        self.attn_hidden_dim = attn_hidden_dim

        # Linear projections for query, key, value
        self.query = nn.Linear(head_dim * head_num, head_dim * head_num)
        self.key = nn.Linear(head_dim * head_num, head_dim * head_num)
        self.value = nn.Linear(head_dim * head_num, head_dim * head_num)

        # Additive attention components
        self.attn_fc = nn.Linear(head_dim * 2, attn_hidden_dim)  # Combine query and key
        self.attn_v = nn.Parameter(torch.randn(attn_hidden_dim))  # Scoring weight vector

        # Final projection layer (initialize as None)
        self.fc = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.size()
        #print(f"Input to MultiHeadAdditiveAttention: {x.shape}")  # Debug

        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.head_num, self.head_dim)  # [B, T, H, D]
        k = self.key(x).view(batch_size, seq_len, self.head_num, self.head_dim)  # [B, T, H, D]
        v = self.value(x).view(batch_size, seq_len, self.head_num, self.head_dim)  # [B, T, H, D]

        #print(f"Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")  # Debug

        # Expand dimensions for queries and keys
        q = q.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)  # [B, T, T, H, D]
        k = k.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)  # [B, T, T, H, D]

        #print(f"Expanded Query shape: {q.shape}, Expanded Key shape: {k.shape}")  # Debug

        # Concatenate Query and Key
        qk_concat = torch.cat((q, k), dim=-1)  # [B, T, T, H, 2D]
        #print(f"Concatenated Query-Key shape: {qk_concat.shape}")  # Debug

        # Compute attention scores
        attn_scores = torch.tanh(self.attn_fc(qk_concat))  # [B, T, T, H, hidden_dim]
        attn_scores = torch.matmul(attn_scores, self.attn_v)  # [B, T, T, H]
        #print(f"Attention scores shape: {attn_scores.shape}")  # Debug

        # Normalize scores
        attn_weights = F.softmax(attn_scores, dim=2)  # [B, T, T, H]
        #print(f"Attention weights shape: {attn_weights.shape}")  # Debug

        # Compute weighted sum of values
        context = torch.matmul(attn_weights, v)  # [B, T, H, D]
        #print(f"Context shape after matmul: {context.shape}")  # Debug

        # Reshape context to combine heads
        context = context.contiguous().view(batch_size, seq_len, -1)  # [B, T, H*D]

        # Dynamic projection layer
        if self.fc is None:
            self.fc = nn.Linear(context.size(-1), embedding_dim).to(x.device)

        #print(f"Final output shape of MultiHeadAdditiveAttention: {context.shape}")  # Debug
        return self.fc(context)


class NewsEncoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, pretrained_embedding=None, dropout=0.2, head_num=16, head_dim=16, attn_hidden_dim=200, seed=None):
        super().__init__()
        torch.manual_seed(seed)

        # Embedding layer
        if pretrained_embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embedding), freeze=False)

        # Projection layer
        if embedding_dim != head_num * head_dim:
            self.projection = nn.Linear(embedding_dim, head_num * head_dim)
        else:
            self.projection = None

        self.dropout = nn.Dropout(dropout)

        self.self_attention = MultiHeadAdditiveAttention(head_num, head_dim, attn_hidden_dim, dropout, seed)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
       # print(f"Input to NewsEncoder: {x.shape}")  # Debug
        x = self.embedding(x)
        #print(f"Shape after embedding: {x.shape}")  # Debug

        if self.projection is not None:
            batch_size, seq_len, _ = x.size()
            x = self.projection(x.view(-1, x.size(-1))).view(batch_size, seq_len, -1)
           # print(f"Shape after projection: {x.shape}")  # Debug

        x = self.dropout(x)
        x = self.self_attention(x)
        #print(f"Shape after self-attention in NewsEncoder: {x.shape}")  # Debug

        x = x.transpose(1, 2)
        x = self.maxpool(x).squeeze(-1)
        #print(f"Output of NewsEncoder after max-pooling: {x.shape}")  # Debug
        return x


class UserEncoder(nn.Module):
    def __init__(self, news_encoder, history_size, head_num=16, head_dim=16, attn_hidden_dim=200, dropout=0.2, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.news_encoder = news_encoder

        self.self_attention = MultiHeadAdditiveAttention(head_num, head_dim, attn_hidden_dim, dropout, seed)

        # Replace AttentionLayer with MeanPooling for simplicity
        self.meanpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, history_titles):
        batch_size, history_size, title_size = history_titles.size()
        #print(f"Input to UserEncoder: {history_titles.shape}")  # Debug

        news_repr = torch.stack([self.news_encoder(history_titles[:, i]) for i in range(history_size)], dim=1)
        #print(f"Shape of news representations: {news_repr.shape}")  # Debug

        news_repr = self.self_attention(news_repr)
        #print(f"Shape after self-attention in UserEncoder: {news_repr.shape}")  # Debug

        news_repr = news_repr.transpose(1, 2)
        user_repr = self.meanpool(news_repr).squeeze(-1)
        #print(f"Output of UserEncoder: {user_repr.shape}")  # Debug
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
            head_num=hparams["head_num"],
            head_dim=hparams["head_dim"],
            attn_hidden_dim=hparams["attention_hidden_dim"],
            dropout=hparams["dropout"],
            seed=seed,
        )

    def forward(self, history_titles, candidate_titles):
        candidate_repr = torch.stack([self.news_encoder(candidate_titles[:, i]) for i in range(candidate_titles.size(1))], dim=1)
        user_repr = self.user_encoder(history_titles)
        
        # Click probability
        scores = torch.bmm(candidate_repr, user_repr.unsqueeze(-1)).squeeze(-1)
        return scores



class HParams:
    def __init__(self):
        self.title_size = 20
        self.history_size = 20
        self.vocab_size = 250002         # lLAMA3 128256 // munin-7b-alpha = 32000 // FastText = 200.000 (2.000.000) // Face -roberta = 250002 // GLOVE = 400004
        self.word_emb_dim = 768 
        self.head_num = 24               #  16 heads (Article)
        self.head_dim = 32               #  16 dim (Article)
        self.attention_hidden_dim = 200  # Artice 200 - additive attention query vectors
        self.dropout = 0.4
        self.learning_rate = 1e-4

    def __getitem__(self, key):
        return getattr(self, key)
