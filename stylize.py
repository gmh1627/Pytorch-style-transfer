import os
import torch
from torchvision import transforms
from transform_pytorch import TransformNet
from utils import load_image, save_image

def stylize(model_path, input_folder, output_folder, image_size=(256, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    transform_net = TransformNet().to(device)
    transform_net.load_state_dict(torch.load(model_path, weights_only=True))
    transform_net.eval()

    # 定义图像转换
    transform = transforms.Compose([
        #transforms.Resize(image_size),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图像
    for img_name in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, img_name)
        output_image_path = os.path.join(output_folder, img_name)

        # 加载并转换图像
        image = load_image(input_image_path)
        image = transform(image).unsqueeze(0).to(device)

        # 进行风格转换
        with torch.no_grad():
            output = transform_net(image).cpu()

        # 保存风格化后的图像
        save_image(output.squeeze(), output_image_path)
        print(f"Saved stylized image to {output_image_path}")

if __name__ == "__main__":
    model_path = "/home/ustcgmh/picture_style_transform/models/epoch_85_starry-night-van-gogh.pth"
    input_folder = "/home/ustcgmh/picture_style_transform/input/"
    output_folder = "/home/ustcgmh/picture_style_transform/output/"
    stylize(model_path, input_folder, output_folder)