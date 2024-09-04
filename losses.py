import torch

def gram_matrix(y):
    """
    计算输入特征图的 Gram 矩阵。
    
    Gram 矩阵用于捕捉图像的风格信息，通过计算特征图的内积来衡量特征之间的相关性。
    
    参数:
    y (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
    
    返回:
    torch.Tensor: Gram 矩阵，形状为 (batch_size, channels, channels)
    """
    (b, ch, h, w) = y.size()  # 获取输入特征图的形状
    features = y.view(b, ch, w * h)  # 将特征图展平为 (batch_size, channels, height * width)
    features_t = features.transpose(1, 2)  # 转置特征图，形状变为 (batch_size, height * width, channels)
    gram = features.bmm(features_t) / (ch * h * w)  # 计算 Gram 矩阵，并进行归一化
    return gram

def content_loss(content_weight, content_features, target_features):
    """
    计算内容损失。
    
    内容损失用于衡量生成图像与内容图像在特征空间中的差异。
    
    参数:
    content_weight (float): 内容损失的权重
    content_features (torch.Tensor): 内容图像的特征图
    target_features (torch.Tensor): 生成图像的特征图
    
    返回:
    torch.Tensor: 内容损失
    """
    return content_weight * torch.mean((target_features - content_features) ** 2)  # 计算均方误差，并乘以权重

def style_loss(style_weight, style_grams, target_features):
    """
    计算风格损失。
    
    风格损失用于衡量生成图像与风格图像在 Gram 矩阵空间中的差异。
    
    参数:
    style_weight (float): 风格损失的权重
    style_grams (list of torch.Tensor): 风格图像的 Gram 矩阵列表
    target_features (list of torch.Tensor): 生成图像的特征图列表
    
    返回:
    torch.Tensor: 风格损失
    """
    loss = 0.0
    for target_feature, style_gram in zip(target_features, style_grams):
        target_gram = gram_matrix(target_feature)  # 计算生成图像的 Gram 矩阵
        if target_gram.size() != style_gram.size():
            print(f"Size mismatch: target_gram {target_gram.size()}, style_gram {style_gram.size()}")  # 打印尺寸不匹配的警告
        loss += torch.mean((target_gram - style_gram) ** 2)  # 计算均方误差，并累加到总损失中
    return style_weight * loss  # 返回加权后的风格损失

def total_variation_loss(img, tv_weight):
    """
    计算总变差损失。
    
    总变差损失用于平滑生成图像，减少噪声和伪影。
    
    参数:
    img (torch.Tensor): 生成图像，形状为 (batch_size, channels, height, width)
    tv_weight (float): 总变差损失的权重
    
    返回:
    torch.Tensor: 总变差损失
    """
    batch_size, c, h, w = img.size()  # 获取生成图像的形状
    tv_h = torch.mean((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2)  # 计算垂直方向的总变差
    tv_w = torch.mean((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2)  # 计算水平方向的总变差
    return tv_weight * (tv_h + tv_w)  # 返回加权后的总变差损失