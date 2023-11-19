import sys
sys.path.append('../research_2/detr')

import torch
import datasets.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import cv2
from util.box_ops import box_cxcywh_to_xyxy
import os.path as osp


model_path = "./exps/r50_detr_epoch1500/checkpoint0499.pth"  # 模型权重路径
image_path = './demo/car_head.jpg'  # 测试图像路径
save_path = './demo/' + osp.basename(image_path).split('.')[0] + '_result.jpg'  # 保存结果图像路径
conf_threshold = 0.7  # 置信度阈值
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    # 加载目标检测模型
    model = torch.load(model_path)['model']  # 加载训练好的模型权重
    model.to(device)  # 将模型移动到 CUDA 上以加速推理
    model.eval()  # 将模型设置为评估模式
    return model

def preprocess_image(image_path):
    # 读取测试图像
    test_image = Image.open(image_path).convert("RGB")

    # 进行图像转换
    img, _ = T.RandomResize([800], max_size=1333)(test_image)
    img, _ = T.ToTensor()(img)
    img, _ = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0).to(device)  # 添加批次维度

    # 原图像和resize后图像的宽高
    w1, h1 = test_image.size
    h2, w2 = img.shape[-2:]
    ratio_height, ratio_width = h1 / h2, w1 / w2

    return test_image, img, (w1, h1), (w2, h2), (ratio_width, ratio_height)

def detect_objects(model, image_tensor, threshold=0.5):
    # 使用模型进行目标检测
    with torch.no_grad():
        predictions = model(image_tensor)

    # 提取预测的目标框、类别
    boxes = torch.squeeze(predictions['pred_boxes'])
    scores, indices = torch.squeeze(predictions['pred_logits']).softmax(1)[:, :-1].max(1)    # logits是MLP输出，需要softmax函数转换为概率

    # 设置阈值以过滤低置信度的目标
    filtered_indices = scores > threshold
    scores = scores[filtered_indices]
    indices = indices[filtered_indices]
    boxes = boxes[filtered_indices]

    return boxes, indices, scores

def draw_and_save_results(image, boxes, indices, scores, wh2, ratio_wh, save_path=None):
    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)

    # 创建可绘制对象
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/smc/Meera.ttf', size=80)

    categories = {0: 'off', 1: 'on'}  # 类别字典

    # 在图像上绘制目标框和标签
    for box, index in zip(boxes, indices):
        box = box.cpu() * torch.tensor([wh2[0], wh2[1], wh2[0], wh2[1]], dtype=torch.float32)
        box = box_cxcywh_to_xyxy(box) * torch.tensor([ratio_wh[0], ratio_wh[1], ratio_wh[0], ratio_wh[1]], dtype=torch.float32)
        box = box.numpy()
        color = (255, 0, 0)  # 设置目标框颜色为绿色 (R, G, B)
        draw.rectangle(box, outline=color, width=8)
        draw.text((box[0], box[1] - 90), categories[index.item()], fill=color, font=font)

    # 保存图像
    if save_path:
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image_cv2)

def main():
    model = load_model(model_path)
    test_image, image_tensor, wh1, wh2, ratio_wh = preprocess_image(image_path)
    boxes, indices, scores = detect_objects(model, image_tensor, threshold=conf_threshold)
    draw_and_save_results(test_image, boxes, indices, scores, wh2, ratio_wh, save_path)

    # 可视化图像
    plt.imshow(test_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
