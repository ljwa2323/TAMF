import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSourceModelWithMaskGuidedEmbedding(nn.Module):
    def __init__(self, dynamic_input_dims, mask_input_dims, static_input_dim, num_sources, embedding_dim, time_embedding_dim, output_dim):
        super(MultiSourceModelWithMaskGuidedEmbedding, self).__init__()
        self.num_sources = num_sources
        # 动态数据源编码器
        self.dynamic_encoders = nn.ModuleList([
            nn.Linear(dynamic_input_dims[k], embedding_dim) for k in range(num_sources)
        ])
        # 遮罩矩阵编码器
        self.mask_encoders = nn.ModuleList([
            nn.Linear(mask_input_dims[k], embedding_dim) for k in range(num_sources)
        ])
        # 时间编码器
        self.time_encoders = nn.ModuleList([
            nn.Linear(1, time_embedding_dim) for _ in range(num_sources)
        ])
        self.static_encoder = nn.Linear(static_input_dim, embedding_dim)
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)  # 交叉注意力模块
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )

    def forward(self, dynamic_sources, time_stamps, masks, static_source):
        dynamic_embeddings = []
        mask_embeddings = []
        for k, (source, time_stamp, mask) in enumerate(zip(dynamic_sources, time_stamps, masks)):
            time_embedding = torch.tanh(self.time_encoders[k](time_stamp.unsqueeze(-1)))
            mask_embedding = torch.tanh(self.mask_encoders[k](mask))
            dynamic_embedding = self.dynamic_encoders[k](source) + time_embedding
            dynamic_embeddings.append(dynamic_embedding)
            mask_embeddings.append(mask_embedding)
        
        # 将动态嵌入和遮罩嵌入合并，准备进行多头自注意力处理
        dynamic_embeddings = torch.stack(dynamic_embeddings)
        mask_embeddings = torch.stack(mask_embeddings)
        
        # 应用多头自注意力模块
        dynamic_attn_output, _ = self.mha(dynamic_embeddings, dynamic_embeddings, dynamic_embeddings)
        mask_attn_output, _ = self.mha(mask_embeddings, mask_embeddings, mask_embeddings)
        
        # 应用交叉注意力模块
        cross_attn_output, _ = self.cross_attention(query=dynamic_attn_output, key=mask_attn_output, value=mask_attn_output)
        
        # 平均池化和静态源处理
        avg_pooled = torch.mean(cross_attn_output, dim=0, keepdim=True)
        static_embedding = F.leaky_relu(self.static_encoder(static_source))
        
        # 合并动态和静态嵌入
        combined_embedding = torch.cat([avg_pooled.squeeze(0), static_embedding], dim=-1)
        
        # 最终表示
        final_representation = self.ffn(combined_embedding)

        return final_representation
    
    def contrastive_loss(self, embeddings, labels):
        n = embeddings.size(0)
        cos_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        losses = []
        for i in range(n):
            for k in range(self.num_sources):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        # Positive pair
                        pos_loss = 1 - cos_sim[i, j]
                        losses.append(F.relu(pos_loss + self.psi))
                    else:
                        # Negative pair
                        neg_loss = cos_sim[i, j]
                        losses.append(F.relu(self.psi - neg_loss))
        loss = sum(losses) / len(losses) if losses else torch.tensor(0.0)
        return loss