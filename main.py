# main.py

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
from losses import gram_matrix, content_loss, style_loss, total_variation_loss
from vgg_network_pytorch import VGG
from transform_pytorch import TransformNet
from utils import load_image

def train(style_image_path, train_path, save_path, epochs=100, batch_size=4, content_weight=1e5, style_weight=1e4, tv_weight=1e-6, learning_rate=0.03):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载风格图像
    style_image = load_image(style_image_path, size=256)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
        #图像通常以浮点数形式表示，像素值范围在 [0, 1] 之间。然而，有时需要将这些值转换回 [0, 255] 的范围，这通常是因为原始图像数据是以 8 位无符号整数（uint8）格式存储的
    ])
    style_image = style_transform(style_image).repeat(batch_size, 1, 1, 1).to(device)

    # 加载 VGG 模型
    vgg = VGG().to(device)

    # 提取风格特征
    style_features = vgg(style_image) # VGG 模型用于提取风格图像的特征，这些特征将用于计算风格损失
    style_grams = [gram_matrix(y) for y in style_features]

    # 初始化风格迁移网络
    transform_net = TransformNet().to(device)
    optimizer = optim.Adam(transform_net.parameters(), lr=learning_rate)

    # 训练的目的是优化风格迁移网络（TransformNet）的参数，使其能够将内容图像转换为具有目标风格的图像
    # 加载训练集
    train_images = [os.path.join(train_path, img) for img in os.listdir(train_path)]
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    for epoch in range(epochs):
        transform_net.train()
        np.random.shuffle(train_images)
        for i in tqdm(range(0, len(train_images), batch_size)):
            batch_images = train_images[i:i+batch_size]
            batch = [train_transform(load_image(img)) for img in batch_images]
            batch = torch.stack(batch).to(device)

            optimizer.zero_grad()
            output = transform_net(batch)

            # 计算损失
            content_features = vgg(batch)
            output_features = vgg(output)
            c_loss = content_loss(content_weight, content_features[2], output_features[2])
            s_loss = style_loss(style_weight, style_grams, output_features)
            tv_loss = total_variation_loss(output, tv_weight)
            loss = c_loss + s_loss + tv_loss

            loss.backward()# 反向传播，计算梯度
            optimizer.step()# 使用优化器更新模型参数

            if (i // batch_size) % 1000 == 0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size}, Loss: {loss.item()}")
            #print(f"Epoch {epoch+1}, Batch {i//batch_size}, Loss: {loss.item()}")

        # 保存模型
        style_name = os.path.splitext(os.path.basename(style_image_path))[0]
        model_filename = f"epoch_{epoch+1}_{style_name}.pth"
        torch.save(transform_net.state_dict(), os.path.join(save_path, model_filename))

    print("训练完成！")

if __name__ == "__main__":
    style_image_path = "/home/ustcgmh/picture_style_transform/examples/starry-night-van-gogh.jpg"
    train_path = "/home/ustcgmh/picture_style_transform/train2014_min"
    save_path = "/home/ustcgmh/picture_style_transform/models"
    os.makedirs(save_path, exist_ok=True)
    train(style_image_path, train_path, save_path)