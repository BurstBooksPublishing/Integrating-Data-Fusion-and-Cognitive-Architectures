# domain_randomizer.py - apply style+noise transforms to RGB and depth
import cv2, numpy as np, sys, os, random
from glob import glob

def randomize(rgb, depth):
    # color jitter
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= random.uniform(0.6,1.2) # saturation
    hsv[...,2] *= random.uniform(0.6,1.2) # value
    rgb = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # gaussian blur
    if random.random()<0.3:
        k = random.choice([3,5,7])
        rgb = cv2.GaussianBlur(rgb,(k,k),0)
    # sensor noise
    noise = np.random.normal(0, random.uniform(2,16), rgb.shape).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16)+noise,0,255).astype(np.uint8)
    # depth dropout and quantization
    d = depth.copy().astype(np.float32)
    mask = np.random.rand(*d.shape) < 0.02
    d[mask] = 0.0 # dropout
    d = np.round(d + np.random.normal(0,0.02*d.max(), d.shape)) # quantize
    return rgb, d.astype(depth.dtype)

if __name__ == "__main__":
    src = sys.argv[1]; dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for f in glob(os.path.join(src,"*.png")):
        rgb = cv2.imread(f)
        depth = cv2.imread(f.replace("rgb","depth"), cv2.IMREAD_ANYDEPTH)
        r,d = randomize(rgb,depth)
        base = os.path.basename(f)
        cv2.imwrite(os.path.join(dst,base), r)          # randomized RGB
        cv2.imwrite(os.path.join(dst,base.replace("rgb","depth")), d) # randomized depth
# Usage: python domain_randomizer.py synth/rgb synth/rgb_rand