import torch
torch.manual_seed(0)
device = "cuda"
torch.cuda.synchronize()

B, N, D = 1, 1024, 1024
ITERS = 50

for dtype, tag in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
    A_g = torch.randn(B*N, D, device=device, dtype=dtype)
    W_g = torch.randn(D,   D, device=device, dtype=dtype)
    A_a = torch.randn(B, N, D, device=device, dtype=dtype)
    B_a = torch.randn(B, N, D, device=device, dtype=dtype)
    A_s = torch.randn(B, N, N, device=device, dtype=dtype)

    # Warm-up (excluded from trace)
    for _ in range(20):
        torch.matmul(A_g, W_g)
        torch.add(A_a, B_a)
        torch.softmax(A_s, dim=-1)
    torch.cuda.synchronize()

    # === NVTX-annotated timed regions ===
    torch.cuda.nvtx.range_push(f"GEMM_{tag}")
    for _ in range(ITERS):
        with torch.no_grad(): torch.matmul(A_g, W_g)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"Add_{tag}")
    for _ in range(ITERS):
        with torch.no_grad(): torch.add(A_a, B_a)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"Softmax_{tag}")
    for _ in range(ITERS):
        with torch.no_grad(): torch.softmax(A_s, dim=-1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

torch.cuda.synchronize()
print("Profiling script complete.")
