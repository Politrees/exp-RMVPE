import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2) 
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))

class DS_Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        c_ = c1
        self.dsconv1 = DSConv(c1, c_, k=3, s=1)
        self.dsconv2 = DSConv(c_, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.dsconv2(self.dsconv1(x)) if self.shortcut else self.dsconv2(self.dsconv1(x))

class DS_C3k(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[DS_Bottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class DS_C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = DS_C3k(c_, c_, n=n, k=k, e=1.0)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        x_ = self.cv1(x)
        x_ = self.m(x_)
        return self.cv2(x_)

class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels, num_hyperedges, num_heads):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)

        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        self.context_mapper = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False)
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        f_avg = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_ctx = torch.cat((f_avg, f_max), dim=1)

        delta_P = self.context_mapper(f_ctx).view(B, self.num_hyperedges, C)
        P = self.global_proto.unsqueeze(0) + delta_P

        z = self.query_proj(x)
        z = z.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        P = P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        sim = (z @ P) * self.scale
        s_bar = sim.mean(dim=1)
        A = F.softmax(s_bar.permute(0, 2, 1), dim=-1)
        return A

class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, A):
        f_m = torch.bmm(A, x) 
        f_m = self.act(self.W_e(f_m))
        x_out = torch.bmm(A.transpose(1, 2), f_m)
        x_out = self.act(self.W_v(x_out))
        return x + x_out

class AdaptiveHypergraphComputation(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges, num_heads):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads)
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        A = self.adaptive_hyperedge_gen(x_flat)
        x_out_flat = self.hypergraph_conv(x_flat, A)
        x_out = x_out_flat.permute(0, 2, 1).view(B, -1, H, W)
        return x_out

class C3AH(nn.Module):
    def __init__(self, c1, c2, num_hyperedges, num_heads, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x_lateral = self.cv1(x)
        x_ahc = self.ahc(self.cv2(x))
        return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))

class HyperACE(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int, 
                 num_hyperedges=16, num_heads=8, k=2, l=1, c_h=0.5, c_l=0.25):
        super().__init__()

        c2, c3, c4, c5 = in_channels 
        c_mid = c4
        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1) 

        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l
        
        self.high_order_branch = nn.ModuleList(
            [C3AH(self.c_h, self.c_h, num_hyperedges=num_hyperedges, num_heads=num_heads, e=1.0) for _ in range(k)]
        )
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)
        
        self.low_order_branch = nn.Sequential(
            *[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)]
        )
        
        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        B2, B3, B4, B5 = x 
        B, _, H4, W4 = B4.shape
        B2_resized = F.interpolate(B2, size=(H4, W4), mode='bilinear', align_corners=False) 
        B3_resized = F.interpolate(B3, size=(H4, W4), mode='bilinear', align_corners=False)
        B5_resized = F.interpolate(B5, size=(H4, W4), mode='bilinear', align_corners=False)

        x_b = self.fuse_conv(torch.cat((B2_resized, B3_resized, B4, B5_resized), dim=1)) 
        x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)

        x_h_outs = [m(x_h) for m in self.high_order_branch]
        x_h_fused = self.high_order_fuse(torch.cat(x_h_outs, dim=1))
        x_l_out = self.low_order_branch(x_l)
        
        y = self.final_fuse(torch.cat((x_h_fused, x_l_out, x_s), dim=1))
        return y

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        return f_in + self.gamma * h