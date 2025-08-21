# -*- coding: utf-8 -*-
"""
评测脚本：生成比赛提交文件 result.txt
- 兼容 YOLOv5 新/旧 attempt_load 签名：device / map_location
- 四通道输入：RGB(3) + IR(1)
- 首行写：参数量(M) 计算量(GFLOPs)
- 其后每行：<可选文件名> [cx cy w h conf cls]...
- 只保留 conf >= 0.25 的框（可调）
"""

import os
import sys
import glob
import yaml
import time
import cv2
import numpy as np
from copy import deepcopy
from pathlib import Path

import torch

# --- YOLOv5 内部依赖 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.general import (
    non_max_suppression, scale_coords, xyxy2xywh, check_img_size, set_logging
)
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def load_yaml(data_yaml):
    with open(data_yaml, 'r', errors='ignore') as f:
        data = yaml.safe_load(f)
    test_vis = data.get('test', None)
    test_ir = data.get('test2', None)
    names = data.get('names', None)
    if test_vis is None or test_ir is None:
        raise FileNotFoundError("YAML 中需包含 test 与 test2 路径。")
    return test_vis, test_ir, names


def list_pairs(vis_dir, ir_dir):
    """按文件名匹配 vis/ir 成对图片；支持 jpg/png/bmp"""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    vis_files = []
    for e in exts:
        vis_files += glob.glob(str(Path(vis_dir) / e))
    vis_files = sorted(vis_files)

    ir_map = {}
    for e in exts:
        for p in glob.glob(str(Path(ir_dir) / e)):
            ir_map[Path(p).stem] = p

    pairs = []
    for v in vis_files:
        stem = Path(v).stem
        if stem in ir_map:
            pairs.append((v, ir_map[stem]))
        else:
            print(f'[WARN] 未找到 IR 配对：{v}')
    return pairs


def percentile_stretch(gray, lower=1.0, upper=99.0):
    """对 IR 做分位拉伸，避免全黑/全白（可通过 --no-ir-stretch 关闭）"""
    lo, hi = np.percentile(gray, (lower, upper))
    if hi - lo < 1e-6:
        return gray.astype(np.float32)
    x = np.clip((gray - lo) / (hi - lo), 0, 1)
    return (x * 255.0).astype(np.float32)


@torch.no_grad()
def compute_params_flops(model, imgsz=640, device='cuda'):
    """
    返回 (参数量M, GFLOPs)。
    为避免 dtype 冲突，统计时强制使用 FP32（.float() + FP32 dummy），失败则切 CPU 兜底。
    """
    n_params = sum(p.numel() for p in model.parameters())
    n_params_m = n_params / 1e6
    gflops = 0.0
    try:
        from thop import profile
        # --- GPU, FP32 ---
        mcopy = deepcopy(model).to(device).eval().float()  # 强制 FP32
        dummy = torch.zeros(1, 4, imgsz, imgsz, device=device, dtype=torch.float32)
        flops, _ = profile(mcopy, inputs=(dummy,), verbose=False)
        gflops = (flops / 1e9) * 2.0  # GMacs->GFLOPs 近似 ×2
        del mcopy, dummy
        torch.cuda.empty_cache()
    except Exception as e:
        # 再试一次：CPU 兜底
        try:
            from thop import profile
            mcopy = deepcopy(model).cpu().eval().float()
            dummy = torch.zeros(1, 4, imgsz, imgsz, device='cpu', dtype=torch.float32)
            flops, _ = profile(mcopy, inputs=(dummy,), verbose=False)
            gflops = (flops / 1e9) * 2.0
        except Exception as e2:
            print(f'[WARN] GFLOPs 统计失败（将写 0.0）：{e2}')
            gflops = 0.0
    return n_params_m, gflops


@torch.no_grad()
def infer_and_write(
    weights,
    data_yaml,
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.60,
    device_str='0',
    save_path='result.txt',
    half=True,
    ir_stretch=True,
    with_names=True,
):
    set_logging()
    device = select_device(device_str)
    half = half and device.type != 'cpu'

    test_vis, test_ir, names = load_yaml(data_yaml)
    pairs = list_pairs(test_vis, test_ir)
    assert len(pairs) > 0, f'未在 {test_vis} / {test_ir} 找到成对图片。'

    # --- 兼容式加载模型 ---
    try:
        model = attempt_load(weights, device=str(device))  # 新版签名
    except TypeError:
        model = attempt_load(weights, map_location=device)  # 旧版签名
    model.eval()

    # stride / imgsz
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
    try:
        imgsz = check_img_size(imgsz, s=stride)
    except TypeError:
        imgsz = check_img_size(imgsz, stride)

    # 先统计 Params & FLOPs（这里内部强制 FP32，不受下方 half 影响）
    params_m, gflops = compute_params_flops(model, imgsz=imgsz, device=device)

    if half:
        model.half()

    # 输出文件
    out_f = open(save_path, 'w', encoding='utf-8')
    out_f.write(f'{params_m:.6f} {gflops:.6f}\n')

    t0 = time.time()
    for idx, (p_vis, p_ir) in enumerate(pairs):
        imv = cv2.imread(p_vis, cv2.IMREAD_COLOR)   # BGR
        imi = cv2.imread(p_ir,  cv2.IMREAD_GRAYSCALE)
        assert imv is not None and imi is not None, f'读取失败：{p_vis} / {p_ir}'

        h0, w0 = imv.shape[:2]

        # IR 预处理
        imi = percentile_stretch(imi) if ir_stretch else imi.astype(np.float32)

        # letterbox（尽量传 stride，兼容旧签名）
        try:
            imv_lb, ratio, pad = letterbox(imv, new_shape=imgsz, auto=False,
                                           scaleFill=False, scaleup=True, stride=stride)
        except TypeError:
            imv_lb, ratio, pad = letterbox(imv, new_shape=imgsz, auto=False,
                                           scaleFill=False, scaleup=True)
        new_unpad = (int(round(w0 * ratio[0])), int(round(h0 * ratio[1])))
        imi_rs = cv2.resize(imi, new_unpad, interpolation=cv2.INTER_LINEAR)
        top = int(round(pad[1])); left = int(round(pad[0]))
        imi_lb = cv2.copyMakeBorder(
            imi_rs,
            top, imgsz - new_unpad[1] - top,
            left, imgsz - new_unpad[0] - left,
            cv2.BORDER_CONSTANT, value=0
        )

        # 拼 4 通道：RGB + IR
        imv_rgb = imv_lb[..., ::-1]  # BGR->RGB
        if imi_lb.ndim == 2:
            imi_lb = imi_lb[..., None]
        im4 = np.concatenate([imv_rgb.astype(np.float32), imi_lb.astype(np.float32)], axis=2)  # HWC,4

        # to tensor
        im = im4.transpose(2, 0, 1)  # CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255.0
        im = im.unsqueeze(0)  # [1,4,H,W]

        # 推理
        pred = model(im, augment=False)[0]  # [N, 6]
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres,
                                   classes=None, agnostic=False)

        # 组装一行
        parts = []
        if with_names:
            parts.append(Path(p_vis).name)  # 写文件名；若平台不需要可 --no-names 关闭

        if len(pred) and len(pred[0]):
            det = pred[0]
            # 反映射到原图尺寸
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], (h0, w0)).round()
            # xyxy -> 归一化 cx,cy,w,h
            gn = torch.tensor([w0, h0, w0, h0], dtype=det.dtype, device=det.device)
            xywh = xyxy2xywh(det[:, :4]) / gn
            confs = det[:, 4]
            clses = det[:, 5].to(torch.int64)

            for j in range(det.shape[0]):
                if confs[j] < conf_thres:
                    continue
                cx, cy, w, h = xywh[j].tolist()
                conf = float(confs[j].item())
                cls_id = int(clses[j].item())
                parts += [f'{cx:.6f}', f'{cy:.6f}', f'{w:.6f}', f'{h:.6f}', f'{conf:.6f}', str(cls_id)]

        out_f.write(' '.join(parts) + '\n')

        if (idx + 1) % 200 == 0:
            print(f'[{idx + 1}/{len(pairs)}] done, elapsed {time.time() - t0:.1f}s')

    out_f.close()
    print(f'✅ 完成。提交文件已生成：{save_path}')
    print(f'首行（参数量, GFLOPs）= {params_m:.6f} {gflops:.6f}')
    return save_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt', help='模型权重路径')
    parser.add_argument('--data', type=str, default='train_file/train_file.yaml', help='数据yaml（含 test/test2）')
    parser.add_argument('--imgsz', type=int, default=640, help='推理分辨率（与训练一致；若训练704就设704）')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值（比赛≥0.25）')
    parser.add_argument('--iou-thres', type=float, default=0.60, help='NMS IoU 阈值')
    parser.add_argument('--device', type=str, default='0', help='cuda设备，如 "0" 或 "cpu"')
    parser.add_argument('--save-path', type=str, default='result.txt', help='输出文件路径')
    parser.add_argument('--no-half', action='store_true', help='禁用FP16推理')
    parser.add_argument('--no-ir-stretch', action='store_true', help='禁用IR分位拉伸')
    parser.add_argument('--no-names', action='store_true', help='每行不写文件名（有的平台需要纯数字行）')
    args = parser.parse_args()

    infer_and_write(
        weights=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device_str=args.device,
        save_path=args.save_path,
        half=(not args.no_half),
        ir_stretch=(not args.no_ir_stretch),
        with_names=(not args.no_names),
    )
