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
        means=[];variances=[];range_variances=[]; residual=[]
        operator_samples=[];observable_masks=[]
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
            operator_samples.append(stack);observable_masks.append(observable)
            mean=stack.mean(1);means.append(mean);variances.append(stack.var(1,unbiased=False))
            range_delta=torch.fft.ifft2(torch.fft.fft2(stack-mean[:,None])*observable).real
            range_variances.append(range_delta.square().mean(1))
            residual.append(torch.fft.ifft2(torch.fft.fft2(mean)*transfer).real)
        means=torch.stack(means,1); within=torch.stack(variances,1)
        if self.c['variant']=='point_operator': weights=F.one_hot(weights.argmax(-1),weights.shape[-1]).float()
        mean=(means*weights[:,:,None,None,None]).sum(1)
        null_variance=(weights[:,:,None,None,None]*within).sum(1)
        operator_variance=(weights[:,:,None,None,None]*(means-mean[:,None]).square()).sum(1)
        range_variance=(weights[:,:,None,None,None]*torch.stack(range_variances,1)).sum(1)
        variance=null_variance+operator_variance
        if self.c['variant']=='baseline': mean=self.prior(h);variance=torch.zeros_like(mean)
        radius=1.645*(variance+1e-6).sqrt()
        return {'prediction':mean,'lower':mean-radius,'upper':mean+radius,'variance':variance,
                'null_variance':null_variance,'operator_variance':operator_variance,'range_variance':range_variance,
                'operator_samples':torch.stack(operator_samples,1),
                'operator_observable':torch.stack(observable_masks)[None].expand(len(x),-1,-1,-1),
                'operator_means':means,'operator_probability':weights,'remeasurement':(torch.stack(residual,1)*weights[:,:,None,None,None]).sum(1)}


    def calibration_scores(self,b,p):
        """Calibrate one global sample scale using only projected null residuals."""
        truth=b['y'].float();height,width=truth.shape[-2:]
        fy=torch.fft.fftfreq(height,device=truth.device)[:,None]
        fx=torch.fft.fftfreq(width,device=truth.device)[None]
        residuals=[]
        for index,sigma in enumerate(self.c.get('sigma',[.7,1.2,2.])):
            transfer=torch.exp(-2*math.pi**2*sigma**2*(fy**2+fx**2))
            observable=(transfer>=self.c.get('threshold',.08)).float()
            if self.c['variant']=='no_projection':observable=torch.zeros_like(observable)
            residual=truth-p['operator_means'][:,index]
            residuals.append(torch.fft.ifft2(torch.fft.fft2(residual)*(1-observable)).real.abs())
        marginalized=(torch.stack(residuals,1)*p['operator_probability'][:,:,None,None,None]).sum(1)
        radius=1.645*p['null_variance'].clamp_min(0).sqrt()
        ratio=torch.where(radius>1e-8,marginalized/radius.clamp_min(1e-8),
                          torch.where(marginalized>1e-8,torch.full_like(radius,float('inf')),torch.zeros_like(radius)))
        return ratio.flatten(1).amax(1)

    calibration_mode='null_sample_scale'

    def apply_calibration(self,p,scale):
        """Scale complete operator-null deviations; preserve observable samples."""
        if not math.isfinite(scale) or scale<0:
            raise ValueError('Finite null-space calibration requires enough independent calibration groups and nonzero posterior support')
        scale=max(1.,scale)
        centered=p['operator_samples']-p['operator_means'][:,:,None]
        samples=p['operator_means'][:,:,None]+scale*centered
        weights=p['operator_probability'][:,:,None,None,None]
        null_variance=(weights*samples.var(2,unbiased=False)).sum(1)
        observable=p['operator_observable'][:,:,None,None]
        range_delta=torch.fft.ifft2(torch.fft.fft2(scale*centered)*observable).real
        range_variance=(weights*range_delta.square().mean(2)).sum(1)
        variance=null_variance+p['operator_variance']
        radius=1.645*(variance+1e-6).sqrt()
        return {**p,'operator_samples':samples,'null_variance':null_variance,
                'range_variance':range_variance,'variance':variance,
                'lower':p['prediction']-radius,'upper':p['prediction']+radius}

    def objective(self,b,p):
        entropy=-(p['operator_probability']*(p['operator_probability']+1e-8).log()).sum(-1).mean()
        error=(p['prediction']-b['y'].float()).square()
        uncertainty=(error/(p['variance']+1e-3)+(p['variance']+1e-3).log()).mean()
        return error.mean()+.2*F.l1_loss(p['remeasurement'],b['x'].float())+.01*uncertainty-.001*entropy


def demo(seed,n,c):
    import numpy as np
    d=image_demo(seed,n,mask=False,size=c.get('input_shape',[3,64,64])[-1])
    height,width=d['y'].shape[-2:]
    fy=np.fft.fftfreq(height)[:,None];fx=np.fft.fftfreq(width)[None]
    sigma=np.random.default_rng(seed+1).choice(c.get('sigma',[.7,1.2,2.]),n)
    transfer=np.exp(-2*np.pi**2*sigma[:,None,None]**2*(fy**2+fx**2))
    d['x']=np.fft.ifft2(np.fft.fft2(d['y'])*transfer[:,None]).real.astype('float32')
    d['operator_sigma']=sigma.astype('float32')
    return d


def measure(p,d,c):
    import numpy as np
    return {'range_violation':float(np.mean((p['remeasurement']-d['x'])**2)),
            'posterior_variance':float(p['variance'].mean()),'null_variance':float(p['null_variance'].mean()),
            'operator_variance':float(p['operator_variance'].mean()),'range_variance':float(p['range_variance'].mean())}
