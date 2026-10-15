import math
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,image_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        w=c.get('width',32)
        self.encoder=Encoder(3,w)
        self.operator=nn.Linear(w,len(c.get('sigma',[.7,1.2,2.])))
        self.prior=nn.Conv2d(w,3,3,padding=1)
        self.scale=nn.Conv2d(w,3,1)

    def forward(self,b):
        x=b['x'].float(); h=self.encoder(x)
        weights=self.operator(h.mean((2,3))).softmax(-1)
        height,width=x.shape[-2:]
        fy=torch.fft.fftfreq(height,device=x.device)[:,None]
        fx=torch.fft.fftfreq(width,device=x.device)[None]
        means=[];variances=[]; residual=[]
        for sigma in self.c.get('sigma',[.7,1.2,2.]):
            transfer=torch.exp(-2*math.pi**2*sigma**2*(fy**2+fx**2))
            observable=(transfer>=self.c.get('threshold',.08)).float()
            if self.c['variant']=='no_projection': observable=torch.zeros_like(observable)
            inverse=torch.where(observable>0,1/transfer.clamp_min(1e-6),torch.zeros_like(transfer))
            base=torch.fft.ifft2(torch.fft.fft2(x)*inverse).real
            samples=[]
            for k in range(self.c.get('samples',6)):
                phase=torch.arange(height*width,device=x.device).reshape(height,width)
                noise=torch.sin(phase*(k+1)*1.618)[None,None]
                prior=self.prior(h)+F.softplus(self.scale(h))*noise
                null=torch.fft.ifft2(torch.fft.fft2(prior)*(1-observable)).real
                samples.append(base+null)
            stack=torch.stack(samples,1)
            mean=stack.mean(1);means.append(mean);variances.append(stack.var(1,unbiased=False))
            residual.append(torch.fft.ifft2(torch.fft.fft2(mean)*transfer).real)
        means=torch.stack(means,1); within=torch.stack(variances,1)
        if self.c['variant']=='point_operator': weights=F.one_hot(weights.argmax(-1),weights.shape[-1]).float()
        mean=(means*weights[:,:,None,None,None]).sum(1)
        variance=(weights[:,:,None,None,None]*(within+(means-mean[:,None]).square())).sum(1)
        if self.c['variant']=='baseline': mean=self.prior(h);variance=torch.zeros_like(mean)
        radius=1.645*(variance+1e-6).sqrt()
        return {'prediction':mean,'lower':mean-radius,'upper':mean+radius,'variance':variance,
                'operator_probability':weights,'remeasurement':(torch.stack(residual,1)*weights[:,:,None,None,None]).sum(1)}

    def objective(self,b,p):
        entropy=-(p['operator_probability']*(p['operator_probability']+1e-8).log()).sum(-1).mean()
        error=(p['prediction']-b['y'].float()).square()
        uncertainty=(error/(p['variance']+1e-3)+(p['variance']+1e-3).log()).mean()
        return error.mean()+.2*F.l1_loss(p['remeasurement'],b['x'].float())+.01*uncertainty-.001*entropy


def demo(seed,n,c):
    from scipy.ndimage import gaussian_filter
    d=image_demo(seed,n,mask=False);d['x']=gaussian_filter(d['y'],(0,0,1.2,1.2));return d


def measure(p,d,c):
    import numpy as np
    return {'range_violation':float(np.mean((p['remeasurement']-d['x'])**2)),
            'posterior_variance':float(p['variance'].mean())}
