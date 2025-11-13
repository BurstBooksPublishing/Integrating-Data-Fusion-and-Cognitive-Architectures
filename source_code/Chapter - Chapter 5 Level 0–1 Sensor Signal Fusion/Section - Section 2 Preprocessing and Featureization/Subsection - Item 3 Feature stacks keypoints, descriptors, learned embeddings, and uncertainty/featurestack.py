import cv2, numpy as np, torch, torchvision.transforms as T
from torch import nn

# Small embedding net with dropout for MC uncertainty.
class EmbedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p=0.2), nn.Linear(32,64)
        )
    def forward(self,x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbedNet().to(device).eval()  # keep eval but enable dropout for MC
# function to enable dropout at inference
def enable_dropout(m): 
    if isinstance(m, nn.Dropout): m.train()
model.apply(enable_dropout)

transform = T.Compose([T.ToPILImage(), T.Resize((64,64)), T.ToTensor()])

img = cv2.imread("frame.jpg")[:,:,::-1]  # BGR->RGB
orb = cv2.ORB_create(500)
kps = orb.detect(img, None)

features = []
for kp in kps:
    x,y = map(int, kp.pt)
    s = int(max(16, kp.size*1.5))
    patch = img[max(0,y-s):y+s, max(0,x-s):x+s]
    if patch.size==0: continue
    t = transform(patch).unsqueeze(0).to(device)
    # MC dropout forward passes
    Tpasses = 8
    reps = [model(t).detach().cpu().numpy().ravel() for _ in range(Tpasses)]
    reps = np.stack(reps, axis=0)
    mean = reps.mean(0); cov = np.cov(reps, rowvar=False) + 1e-6*np.eye(reps.shape[1])
    # Hamming descriptor via ORB
    kp2 = cv2.KeyPoint(x,y,kp.size,kp.angle)
    _, des = orb.compute(img, [kp2])
    features.append({
        "sensor":"cam0","t":0.0,"pt":(x,y),"scale":kp.size,"angle":kp.angle,
        "descriptor": None if des is None else des[0],
        "embedding_mean": mean, "embedding_cov": cov,
        "detector_score": kp.response
    })
# features ready for storage/association