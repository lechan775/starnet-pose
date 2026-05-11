"""测量模型推理延迟"""
import torch
import time
import argparse
from mmpose.apis.inference import init_model

def benchmark_latency(config, checkpoint=None, device='cuda:0', warmup=50, runs=200):
    """测量模型推理延迟"""
    # 初始化模型
    model = init_model(config, checkpoint=checkpoint, device=device)
    model.eval()
    
    # 创建输入
    input_shape = (1, 3, 256, 192)
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model._forward(dummy_input)
    
    # 同步 CUDA
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # 测量延迟
    print(f"Benchmarking ({runs} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model._forward(dummy_input)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转换为毫秒
    
    # 计算统计
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    fps = 1000 / avg_latency
    
    return {
        'avg_latency_ms': avg_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'fps': fps
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--runs', type=int, default=200)
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"{'='*50}\n")
    
    results = benchmark_latency(
        args.config, 
        args.checkpoint, 
        args.device,
        args.warmup,
        args.runs
    )
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Average Latency: {results['avg_latency_ms']:.2f} ms")
    print(f"  Min Latency: {results['min_latency_ms']:.2f} ms")
    print(f"  Max Latency: {results['max_latency_ms']:.2f} ms")
    print(f"  FPS: {results['fps']:.1f}")
    print(f"{'='*50}\n")
