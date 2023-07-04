#  CSDA
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        # self.act=nn.SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)
class SpatialAttentionModule(nn.Module):
    def __init__(self, c1):
        super(SpatialAttentionModule, self).__init__()
        # self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        reduce = 8
        c_ = int(c1 // reduce)
        self.cv1 = nn.Conv2d(in_channels=c1, out_channels=c_, kernel_size=1, stride=1,padding=0)
        self.dc1 = nn.Conv2d(c_, c_, kernel_size=(3,3), dilation=(2,2), stride=(1,1), padding=2)
        self.dc2 = nn.Conv2d(c_, c_, kernel_size=(3,3), dilation=(2,2), stride=(1,1), padding=2)
        self.cv2 = nn.Conv2d(in_channels=c_, out_channels=c1, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # out = self.act(self.conv2d(out))
        out = self.cv1(x)
        out = self.dc1(out)
        out = self.dc2(out)
        out = self.cv2(out)
        return self.act(out)
class CSDA(nn.Module):
    def __init__(self, c1, c2):
        super(CSDA, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule(c1)

    def forward(self, x):
        out = self.spatial_attention(x) + x
        out = self.channel_attention(out) * out
        return out


#  DWELAN
class DWBottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,5), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=c_)  #self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.dwConv = nn.Conv2d(c_, c_, kernel_size=k[1],padding=autopad(k[1]),groups=c_)
        self.cv2 = Conv(c_,c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv1(x)
        out = self.dwConv(out)
        out = self.cv2(out)
        return x + out if self.add else out
class DWBottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,5), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=c_)  #self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.dwConv = nn.Conv2d(c_, c_, kernel_size=k[1],padding=autopad(k[1]),groups=c_)
        self.cv2 = Conv(c_,c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv1(x)
        out = self.dwConv(out)
        out = self.cv2(out)
        return x + out if self.add else out
class DWELAN(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DWBottleneck(self.c, self.c, shortcut, g, k=(3, 5), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



#  NWD-LOSS
    pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
    pxy = pxy.sigmoid() * 2 - 0.5
    pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
    pbox = torch.cat((pxy, pwh), 1)  # predicted box

    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
    selected_tbox = tbox[i].T
    b1_cx, b1_cy, b1_w, b1_h = pbox.T[0], pbox.T[1], pbox.T[2], pbox.T[3]
    b2_cx, b2_cy, b2_w, b2_h = selected_tbox[0], selected_tbox[1], selected_tbox[2], selected_tbox[3]
    cx_L2Norm = torch.pow((b1_cx - b2_cx), 2)
    cy_L2Norm = torch.pow((b1_cy - b2_cy), 2)
    p1 = cx_L2Norm + cy_L2Norm
    w_FroNorm = torch.pow((b1_w - b2_w) / 2, 2)
    h_FroNorm = torch.pow((b1_h - b2_h) / 2, 2)
    p2 = w_FroNorm + h_FroNorm
    wasserstein = torch.exp(-torch.pow((p1 + p2), 1 / 2) / 2.5)
    lbox += (1.0 - wasserstein).mean()