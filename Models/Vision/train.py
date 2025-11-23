import os
import torch
import pandas as pd
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch.optim as optim

class SleepersDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, images_dir, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transforms = transforms
        self.imgs = self.data['filename'].unique()[:100]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # Aplicar transform **aquí**
        if self.transforms:
            img = self.transforms(img)

        records = self.data[self.data['filename'] == img_name]
        boxes = records[['xmin','ymin','xmax','ymax']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float16)
        labels = torch.ones((len(records),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return img, target


# Cargar Faster R-CNN preentrenado
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')



# Reemplazar la cabeza de clasificación (solo 1 clase: sleeper)
num_classes = 2  # 0=background, 1=sleeper
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)

transform = T.Compose([
    T.ToTensor(),             # PIL → Tensor
    T.Resize((180,320))
])


dataset = SleepersDataset("Dataset/train/_annotations.csv", "Dataset/train/")
print("Dataset Cargado")
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 20

transform = T.Compose([
    T.ToTensor(), 
])

for epoch in range(num_epochs):
    model.train()
    for imgs, targets in data_loader:
        imgs = [transform(img).to(device) for img in imgs]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {losses.item():.4f}")


torch.save(model.state_dict(), "fasterrcnn_sleeper.pth")


