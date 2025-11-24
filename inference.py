import torch
import os
import os.path as osp
import argparse

from PIL import Image
from torchvision import transforms

from basicsr.archs.efmsr_arch import EFMSR

model_path = {
    "classical": {
        "2": "experiments/pretrained_models/EFMSR_x2.pth",
        "3": "experiments/pretrained_models/EFMSR_x3.pth",
        "4": "experiments/pretrained_models/EFMSR_x4.pth",
    },
    "lightweight": {
        "2": "experiments/pretrained_models/EFMSR_Light_x2.pth",
        "3": "experiments/pretrained_models/EFMSR_Light_x3.pth",
        "4": "experiments/pretrained_models/EFMSR_Light_x4.pth",
    }
}


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input image or directory path.")
    parser.add_argument("-o", "--out_path", type=str, default="results/test/", help="Output directory path.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument(
        "--task",
        type=str,
        default="classical",
        choices=['classical', 'lightweight'],
        help="Task for the model. classical: for classical SR models. lightweight: for lightweight models."
    )
    args = parser.parse_args()

    return args


def process_image(image_input_path, image_output_path, model, device):
    with torch.no_grad():
        image_input = Image.open(image_input_path).convert('RGB')
        image_input = transforms.ToTensor()(image_input).unsqueeze(0).to(device)
        image_output = model(image_input).clamp(0.0, 1.0)[0].cpu()
        image_output = transforms.ToPILImage()(image_output)
        image_output.save(image_output_path)


def main():
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    global model

    if args.task == 'classical':
        model = EFMSR(upscale=args.scale)
    elif args.task == 'lightweight':
        model = EFMSR(
            upscale=args.scale,
            embed_dim=48,
            depths=(1, 2, 3, 3, 3),
            upsampler='pixel_shuffle_direct',
            num_heads=(4, 4, 4, 4, 4),
            num_topk=(
                1024,
                256, 256,
                128, 128, 128,
                64, 64, 64,
                32, 32, 32,
            )
        )

    state_dict = torch.load(model_path[args.task][str(args.scale)], map_location=device)['params_ema']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if os.path.isdir(args.in_path):
        for file in os.listdir(args.in_path):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                image_input_path = osp.join(args.in_path, file)
                image_output_path = os.path.join(args.out_path, f"EFMSR_x{args.scale}_{file}")
                process_image(image_input_path, image_output_path, model, device)
    else:
        if args.in_path.endswith('.png') or args.in_path.endswith('.jpg') or args.in_path.endswith('.jpeg'):
            image_input_path = args.in_path
            file_name = osp.basename(args.in_path)
            image_output_path = os.path.join(args.out_path, f"EFMSR_x{args.scale}_{file_name}")
            process_image(image_input_path, image_output_path, model, device)


if __name__ == "__main__":
    main()
