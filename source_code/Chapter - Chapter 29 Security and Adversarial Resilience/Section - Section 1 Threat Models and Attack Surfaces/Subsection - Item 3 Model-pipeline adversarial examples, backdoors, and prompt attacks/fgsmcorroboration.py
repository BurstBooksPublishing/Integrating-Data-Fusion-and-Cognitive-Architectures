import torch, torchvision.transforms as T, torchvision.models as models
from PIL import Image
# load model and preprocess
model = models.resnet18(pretrained=True).eval()
transform = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
img = transform(Image.open("scene.jpg")).unsqueeze(0)  # single image input
img.requires_grad = True
# FGSM step (epsilon small)
logits = model(img)
target = logits.argmax(dim=1)  # simulate misclassification target selection
loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()
epsilon = 0.01
adv_img = img + epsilon * img.grad.sign()
adv_img = torch.clamp(adv_img, 0, 1)
# compare predictions
orig_pred = model(img).argmax(dim=1).item()
adv_pred  = model(adv_img).argmax(dim=1).item()
# simple cross-modal corroboration: symbolic tag from secondary sensor
secondary_tag = "vehicle"  # e.g., radar/camera fused ID
vision_tag = "vehicle" if orig_pred==adv_pred else "unknown"
# decision rule: require match to act
if vision_tag == secondary_tag:
    print("Proceed with fused action")  # safe-path
else:
    print("Quarantine: discrepancy detected; escalate")  # containment