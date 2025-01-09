import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from datetime import datetime

from model import MultiSourceModel, focal_loss, reconstruction_loss, contrastive_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Time-aware Multi-source EMR Model Training')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--static_dim', type=int, default=None, help='Static feature dimension')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lambda_focal', type=float, default=1.0, help='Weight for focal loss')
    parser.add_argument('--lambda_recon', type=float, default=0.5, help='Weight for reconstruction loss')
    parser.add_argument('--lambda_contrast', type=float, default=0.3, help='Weight for contrastive loss')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--input_dims', nargs='+', type=int, required=True, help='Input dimensions for each source')
    parser.add_argument('--mask_dims', nargs='+', type=int, required=True, help='Mask dimensions for each source')
    parser.add_argument('--time_dims', nargs='+', type=int, required=True, help='Time dimensions for each source')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Print loss every n batches')
    
    return parser.parse_args()

def setup_logging(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_epoch(model, train_loader, optimizer, args):
    model.train()
    total_loss = 0
    
    for batch_idx, (x_list, m_list, t_list, static, target) in enumerate(train_loader):
        # 将数据移到指定设备
        x_list = [x.to(args.device) for x in x_list]
        m_list = [m.to(args.device) for m in m_list]
        t_list = [t.to(args.device) for t in t_list]
        static = static.to(args.device) if static is not None else None
        target = target.to(args.device)
        
        optimizer.zero_grad()
        
        # 前向传播
        classification_output, embeddings, reconstructions = model(x_list, m_list, t_list, static)
        
        # 计算损失
        focal = focal_loss(classification_output, target)
        recon = reconstruction_loss(reconstructions, x_list, m_list)
        pooled_embeddings = [torch.mean(x_out, dim=0) for x_out, _ in embeddings]
        contrast = contrastive_loss(pooled_embeddings)
        
        loss = (args.lambda_focal * focal + 
                args.lambda_recon * recon + 
                args.lambda_contrast * contrast)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % args.log_interval == 0:
            logging.info(f'Train Batch [{batch_idx + 1}/{len(train_loader)}] - '
                        f'Loss: {loss.item():.4f} '
                        f'(Focal: {focal.item():.4f}, '
                        f'Recon: {recon.item():.4f}, '
                        f'Contrast: {contrast.item():.4f})')
    
    return total_loss / len(train_loader)

def main():
    args = parse_args()
    setup_logging(args.save_dir)
    
    logging.info(f"Starting training with arguments: {vars(args)}")
    
    # 初始化模型
    model = MultiSourceModel(
        input_dims=args.input_dims,
        mask_dims=args.mask_dims,
        time_dims=args.time_dims,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        static_dim=args.static_dim
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # TODO: 实现数据加载器
    train_loader = DataLoader(
        # 这里需要实现自定义的Dataset类
        dataset=None,  
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # 训练循环
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        avg_loss = train_epoch(model, train_loader, optimizer, args)
        logging.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main() 