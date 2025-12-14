import argparse
import torch
import torch.nn as nn

class ThresholdModel(nn.Module):
    def __init__(self, thresh=20000.0):
        super().__init__()
        self.thresh = thresh

    def forward(self, x):
        return (x > self.thresh).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=20000.0)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=512)
    args = ap.parse_args()

    model = ThresholdModel(args.thresh)
    model.eval()

    dummy = torch.zeros(1, 1, args.h, args.w, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,   # ★ ここは18でOK
    )

    print("exported:", args.out)

if __name__ == "__main__":
    main()
