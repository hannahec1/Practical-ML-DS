#%%
from torchvision import transforms, models
import torch
from PIL import Image
import torch.nn.functional as F

model = models.detection.retinanet_resnet50_fpn(pretrained=True)

img = Image.open('plane.jpg')
# img.show()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = transform(img)
img = img.unsqueeze(0)

model.eval()
pred = model(img)
print(pred)
#%%
mask = pred[0]['masks'][0]
t = transforms.ToPILImage()
t(mask).show()
#%%
# print(pred.shape)
pred = F.softmax(pred)
print(torch.sum(pred))
pred = torch.argmax(pred, dim=1)
print(pred)
