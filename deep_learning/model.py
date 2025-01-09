import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

# 嵌入层定义
class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TimeEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, t):
        return torch.tanh(self.fc(t))

class MaskGuidedSequenceEmbedding(nn.Module):
    def __init__(self, input_dim, mask_dim, time_dim, embed_dim, num_heads):
        super(MaskGuidedSequenceEmbedding, self).__init__()
        self.time_embedding = TimeEmbedding(time_dim, embed_dim)
        self.value_embedding = nn.Linear(input_dim, embed_dim)
        self.mask_embedding = nn.Linear(mask_dim, embed_dim)
        self.self_attention_x = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attention_m = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x, m, t):
        # 时间嵌入
        time_embed = self.time_embedding(t)
        
        # 输入和掩码嵌入
        x_embed = self.value_embedding(x) + time_embed
        m_embed = self.mask_embedding(m) + time_embed
        
        # 自注意力
        x_out, _ = self.self_attention_x(x_embed, x_embed, x_embed)
        m_out, _ = self.self_attention_m(m_embed, m_embed, m_embed)
        
        # 跨源注意力
        x_out, _ = self.cross_attention(x_out, m_out, x_out)
        
        return x_out, m_out

class CrossSourceFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossSourceFusion, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, embeddings):
        embeddings = torch.cat(embeddings, dim=0)
        out, _ = self.self_attention(embeddings, embeddings, embeddings)
        return out

# 分类器定义
class FinalClassification(nn.Module):
    def __init__(self, input_dim):
        super(FinalClassification, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

# 添加重建解码器
class SequenceDecoder(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads):
        super(SequenceDecoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
    def forward(self, source_embed, fused_embed):
        # 使用融合嵌入作为key，源嵌入作为query和value
        decoded, _ = self.attention(source_embed, fused_embed, source_embed)
        decoded = self.ffn(decoded)
        return self.output_proj(decoded)

# 多源模型定义
class MultiSourceModel(nn.Module):
    def __init__(self, input_dims, mask_dims, time_dims, embed_dim, num_heads, static_dim=None):
        super(MultiSourceModel, self).__init__()
        self.sequence_embeddings = nn.ModuleList([
            MaskGuidedSequenceEmbedding(in_dim, mask_dim, time_dim, embed_dim, num_heads) 
            for in_dim, mask_dim, time_dim in zip(input_dims, mask_dims, time_dims)
        ])
        
        # 添加重建解码器
        self.decoders = nn.ModuleList([
            SequenceDecoder(embed_dim, input_dim, num_heads)
            for input_dim in input_dims
        ])
        
        # 静态特征处理
        if static_dim:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, embed_dim),
                nn.LeakyReLU()
            )
        
        self.cross_source_fusion = CrossSourceFusion(embed_dim, num_heads)
        final_dim = embed_dim + len(input_dims) * embed_dim
        if static_dim:
            final_dim += embed_dim
        self.final_classification = FinalClassification(final_dim)

    def forward(self, x_list, m_list, t_list, static=None):
        # 序列嵌入
        embeddings = [seq_embed(x, m, t) for seq_embed, x, m, t in zip(self.sequence_embeddings, x_list, m_list, t_list)]
        
        # 跨源融合
        fused = self.cross_source_fusion([x_out for x_out, _ in embeddings])
        
        # 重建
        reconstructions = [decoder(x_out, fused) for decoder, (x_out, _) in zip(self.decoders, embeddings)]
        
        # 合并嵌入
        concat_embeddings = [fused] + [x_out for x_out, _ in embeddings]
        if static is not None:
            static_embed = self.static_proj(static)
            concat_embeddings.append(static_embed)
        
        concat_embeddings = torch.cat(concat_embeddings, dim=1)
        classification_output = self.final_classification(concat_embeddings)
        
        return classification_output, embeddings, reconstructions

# 损失函数定义
def focal_loss(pred, target, gamma=2.0, beta=0.5):
    pred = pred.clamp(1e-5, 1 - 1e-5)
    loss = -beta * (1 - pred)**gamma * target * torch.log(pred) - (1 - beta) * pred**gamma * (1 - target) * torch.log(1 - pred)
    return loss.mean()

def reconstruction_loss(reconstructions, x_list, m_list):
    loss = 0.0
    batch_size = x_list[0].shape[0]
    num_sources = len(x_list)
    
    for recon, x, m in zip(reconstructions, x_list, m_list):
        masked_mse = ((recon - x) * m).pow(2).sum() / (m.sum() * x.shape[-1])
        loss += masked_mse
        
    return loss / (batch_size * num_sources)

def contrastive_loss(pooled_embeddings, margin=1.0):
    B = pooled_embeddings[0].shape[0]  # Batch size
    K = len(pooled_embeddings)         # Number of data sources
    loss = 0.0
    # Iterate over each sample in the batch
    for b in range(B):
        # Randomly select a different sample j from the batch, j != b
        j = random.choice([x for x in range(B) if x != b])
        # Iterate over each pair of data sources
        for k in range(K):
            for k_prime in range(K):
                if k != k_prime:
                    # Calculate the distance between embeddings from the same data source for sample b
                    dist_same = F.pairwise_distance(pooled_embeddings[k][b].unsqueeze(0), pooled_embeddings[k_prime][b].unsqueeze(0))
                    # Calculate the distance between embeddings from the same data source for sample b and different sample j
                    dist_diff = F.pairwise_distance(pooled_embeddings[k][b].unsqueeze(0), pooled_embeddings[k_prime][j].unsqueeze(0))
                    loss += F.relu(dist_same - dist_diff + margin).mean()
    # Normalize the loss by the number of comparisons
    return loss / (B * K * (K - 1))

# 训练函数定义
def train_model(model, train_loader, num_epochs=20, lr=0.001, lambda_focal=1.0, lambda_recon=0.5, lambda_contrast=0.3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for x_list, m_list, t_list, target in train_loader:
            optimizer.zero_grad()
            classification_output, embeddings = model(x_list, m_list, t_list)
            loss = focal_loss(classification_output, target)
            reconstructions = [model.reconstruct(x, m, t) for x, m, t in zip(x_list, m_list, t_list)]
            rec_loss = reconstruction_loss(reconstructions, x_list, m_list)
            pooled_embeddings = [torch.mean(x_out, dim=0) for x_out, _ in embeddings]  # Assuming pooling is mean pooling
            cont_loss = contrastive_loss(pooled_embeddings)
            total_loss = lambda_focal * loss + lambda_recon * rec_loss + lambda_contrast * cont_loss
            total_loss.backward()
            optimizer.step()
            total_loss += total_loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")