import cv2
import numpy as np

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
NUM_FRAMES    = 16
BATCH_SIZE    = 8

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def video_crop(video_frame, type):
    l = video_frame.shape[0]
    new_frame = []
    for i in range(l):
        img = cv2.resize(video_frame[i], dsize=(680, 512))
        new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    #1
    img = np.array(new_frame)
    if type == 0:
        img = img[:, 32:480, 116:564, :]
    #2
    elif type == 1:
        img = img[:, :448, :448, :]
    #3
    elif type == 2:
        img = img[:, :448, -448:, :]
    #4
    elif type == 3:
        img = img[:, -448:, :448, :]
    #5
    elif type == 4:
        img = img[:, -448:, -448:, :]
    #6
    elif type == 5:
        img = img[:, 32:480, 116:564, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #7
    elif type == 6:
        img = img[:, :448, :448, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #8
    elif type == 7:
        img = img[:, :448, -448:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #9
    elif type == 8:
        img = img[:, -448:, :448, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #10
    elif type == 9:
        img = img[:, -448:, -448:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)

    return img

def image_crop(image, type):
    img = cv2.resize(image, dsize=(680, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #1
    if type == 0:
        img = img[32:480, 116:564, :]
    #2
    elif type == 1:
        img = img[:448, :448, :]
    #3
    elif type == 2:
        img = img[:448, -448:, :]
    #4
    elif type == 3:
        img = img[-448:, :448, :]
    #5
    elif type == 4:
        img = img[-448:, -448:, :]
    #6
    elif type == 5:
        img = img[32:480, 116:564, :]
        img = cv2.flip(img, 1)
    #7
    elif type == 6:
        img = img[:448, :448, :]
        img = cv2.flip(img, 1)
    #8
    elif type == 7:
        img = img[:448, -448:, :]
        img = cv2.flip(img, 1)
    #9
    elif type == 8:
        img = img[-448:, :448, :]
        img = cv2.flip(img, 1)
    #10
    elif type == 9:
        img = img[-448:, -448:, :]
        img = cv2.flip(img, 1)

    return img

def get_image_features(model, pixel_values, device):
    with torch.no_grad():
        pixel_values = pixel_values.to(torch.bfloat16).to(device)
        vit_embeds = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True
        ).last_hidden_state
        cls_token = vit_embeds[:, 0, :]  # [B, 1024]
    return cls_token

def load_video_frames(video_path, num_frames=NUM_FRAMES):
    """Uniform sampling num_frames từ video, trả về numpy [T, H, W, 3] BGR"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return np.array(frames)  # [T, H, W, 3] BGR

if __name__ == '__main__':
    video_path = 'sample.mp4'

    video_frames = load_video_frames(video_path, num_frames=NUM_FRAMES)
    print(f"Sampled {video_frames.shape[0]} frames")

    corp_video = video_crop(video_frames, 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        "ppxin321/HolmesVAU-2B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    transform = build_transform(input_size=448)

    video_features = torch.zeros(0).to(device)
    with torch.no_grad():
        for i in range(0, corp_video.shape[0], BATCH_SIZE):
            batch = corp_video[i:i + BATCH_SIZE]                              # [B, H, W, 3]
            imgs = torch.stack([transform(Image.fromarray(f)) for f in batch])  # [B, 3, 448, 448]
            feature = get_image_features(model, imgs, device)                 # [B, 1024]
            video_features = torch.cat([video_features, feature], dim=0)

    video_features = video_features.detach().cpu().float().numpy()
    np.save('save_path', video_features)