import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# --- CONFIGURATION ---
DEVICE = torch.device("cpu")
PATH_TO_UNET = 'models/best_unet_model.pth'
PATH_TO_CLASSIFIER = 'models/phase1_resnet_pytorch_final.pth'

# --- 1. U-NET ARCHITECTURE ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

# --- 2. GRAD-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output
            output.register_hook(self.save_gradient)
            
        self.target_layer.register_forward_hook(forward_hook)

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x):
        self.model.zero_grad()
        x.requires_grad_(True)
        output = self.model(x)
        score = output[0, 0]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((x.shape[2], x.shape[3]))
            
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()[0, 0]
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# --- 3. MODEL LOADER ---
unet_model = None
resnet_model = None
grad_cam_tool = None

def load_models():
    global unet_model, resnet_model, grad_cam_tool
    if unet_model and resnet_model: return True

    try:
        unet = UNet(in_channels=3, out_channels=1).to(DEVICE)
        unet.load_state_dict(torch.load(PATH_TO_UNET, map_location=DEVICE))
        unet.eval()
        unet_model = unet

        resnet = models.resnet50(pretrained=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
        resnet.load_state_dict(torch.load(PATH_TO_CLASSIFIER, map_location=DEVICE), strict=False)
        resnet.to(DEVICE)
        resnet.eval()
        resnet_model = resnet
        
        grad_cam_tool = GradCAM(resnet_model, resnet_model.layer4[-1])
        return True
    except Exception as e:
        print(f"Model Load Error: {e}")
        return False

# --- 4. INTEGRATION PIPELINE ---
def run_phase3(image_path):
    if not load_models(): return {"diagnosis": "Error", "color": "black"}

    original = cv2.imread(image_path)
    if original is None: return None
    display_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(display_img, (1024, 1024))
    img_norm = (img_resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(DEVICE).float()

    with torch.no_grad():
        mask_pred = unet_model(img_tensor)
        mask_pred = torch.sigmoid(mask_pred) > 0.05 
        mask_np = mask_pred.squeeze().cpu().numpy().astype(np.uint8)

    mask_full = cv2.resize(mask_np, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resnet_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    blast_count = 0
    healthy_count = 0
    global_heatmap = np.zeros((original.shape[0], original.shape[1]), dtype=np.float32)
    boxes_to_draw = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 20: continue 

        pad = 20
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(original.shape[1], x+w+pad), min(original.shape[0], y+h+pad)
        
        # Prevent math errors if crop goes off-screen
        w_crop, h_crop = x2 - x1, y2 - y1
        if w_crop <= 0 or h_crop <= 0: continue
        
        cell_crop = original[y1:y2, x1:x2]
        if cell_crop.size == 0: continue

        cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
        cell_tensor = resnet_transform(Image.fromarray(cell_rgb)).unsqueeze(0).to(DEVICE)

        with torch.set_grad_enabled(True):
            output_logits = resnet_model(cell_tensor)
            prediction = torch.sigmoid(output_logits).item()

            if prediction > 0.5:
                blast_count += 1
                color = (0, 0, 255) 
                label = "BLAST"
                try:
                    local_heatmap = grad_cam_tool(cell_tensor)
                    local_heatmap_resized = cv2.resize(local_heatmap, (w_crop, h_crop))
                    
                    # --- THE EDGE DESTROYER (Gaussian Window) ---
                    k_x = cv2.getGaussianKernel(w_crop, w_crop / 4)
                    k_y = cv2.getGaussianKernel(h_crop, h_crop / 4)
                    window = k_y @ k_x.T
                    window = window / window.max()
                    
                    softened_heatmap = local_heatmap_resized * window
                    
                    global_heatmap[y1:y2, x1:x2] += (softened_heatmap * prediction)
                except Exception as e:
                    pass
            else:
                healthy_count += 1
                color = (0, 255, 0) 
                label = "Healthy"

        boxes_to_draw.append((x, y, w, h, color, label, prediction))

    # MASSIVE CONTINUOUS BLUR
    global_heatmap = cv2.GaussianBlur(global_heatmap, (151, 151), 0)
    if global_heatmap.max() > 0:
        global_heatmap = global_heatmap / global_heatmap.max()

    # GENERATE COLORS & TRANSPARENCY MASK
    heatmap_color = cv2.applyColorMap(np.uint8(255 * global_heatmap), cv2.COLORMAP_JET)
    alpha = np.clip((global_heatmap - 0.10) * 3.5, 0, 0.65)
    alpha_3d = np.expand_dims(alpha, axis=2)

    # BLEND
    output_image = (original.astype(np.float32) * (1 - alpha_3d) + heatmap_color.astype(np.float32) * alpha_3d).astype(np.uint8)

    # DRAW BOXES
    for box in boxes_to_draw:
        x, y, w, h, color, label, prediction = box
        cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 3)
        cv2.putText(output_image, f"{label} {prediction:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    total = blast_count + healthy_count
    bp = (blast_count / total * 100) if total > 0 else 0
    if bp >= 20.0: diagnosis, d_color = f"POSITIVE ({bp:.1f}%)", "red"
    elif bp > 0: diagnosis, d_color = f"SUSPICIOUS ({bp:.1f}%)", "orange"
    else: diagnosis, d_color = "NEGATIVE", "green"

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    res_path = image_path.replace('uploads', 'results')
    cv2.imwrite(res_path, cv2.cvtColor(output_image_rgb, cv2.COLOR_RGB2BGR))

    return {
        "diagnosis": diagnosis, "color": d_color,
        "blast_count": blast_count, "healthy_count": healthy_count,
        "result_image": res_path
    }