
from torch import nn
import torch
import math

def gaussian_kaiming_uniform_(tensor, gain=0.33, mode='fan_in'):
    
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    # print(std,"std")
    bound = math.sqrt(3.0) * std  # Calculate the bound for the uniform distribution
    # bound=1
    # print(bound)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
 
def get_rbf_width(distance,k):
    return (2/k*1/distance)**2
        
class RBF(nn.Module):
    def __init__(self, in_features, out_features,dtype=None,device=None,ranges=None):
        super(RBF, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        # self.fc=PreactLinear(in_features, out_features*deg,bias=False)
        # self.weight=
        # if centers is None:
        self.weight=nn.Parameter(torch.zeros(in_features,out_features,**factory_kwargs))
        
        # torch.nn.init.kaiming_uniform_(self.weight.T , a= 2.963)
        gaussian_kaiming_uniform_(self.weight.T)
     
        self.centers = nn.Parameter(torch.zeros(in_features,out_features,**factory_kwargs))
        
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weight.T)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        # bound=bound/(self.out_features)
        if ranges is None:
            ranges=[-bound,bound]
        # print(ranges)
        # torch.nn. init.uniform_(self.centers, ranges[0], ranges[1])
        self.centers.data=torch.linspace(ranges[0],ranges[1],out_features,device=self.centers.device,dtype=self.centers.dtype).unsqueeze(0).repeat(in_features,1)
       
      
      
        beta =(get_rbf_width(2/out_features,math.log2(out_features)))
        # print("beta",beta)
        # self.beta=torch.nn.Parameter(torch.ones(in_features,out_features)*beta)
        self.beta=beta
        # self.beta=1
      
  
    def forward(self, x):
        center=self.centers
        x=x.unsqueeze(-1)*self.weight.unsqueeze(0) -center.unsqueeze(0)
        # print(center.shape,self.centers.shape)
        # print(center.shape,x.shape)
        distance=(x**2)
        y= torch.exp(-self.beta * distance)
        # y=torch.exp(-distance/(x.var()**2))
       
        
        
        # print(y.shape,"y.shape")
        return y
    def get_subset(self,in_id,target):
        target.weight.data=self.weight[in_id].clone()
        target.centers.data=self.centers[in_id].clone()
        target.beta=self.beta
        
class KANLayer(nn.Module):
    def __init__(self, in_features,out_features,num_basis=5,base_fn=...,basis_trainable=False,ranges=None,dtype=None,device=None):
        super(KANLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features=in_features
        self.out_features=out_features
        self.num_centers=num_basis
        self.rbf = RBF(in_features, num_basis,ranges=ranges,**factory_kwargs).requires_grad_(basis_trainable)
     
        weight=torch.zeros(out_features,in_features*num_basis,**factory_kwargs)
        # torch.nn.init.kaiming_uniform_(weight, a=2.963)
        gaussian_kaiming_uniform_(weight)
        
        self.weight=nn.Parameter(weight.reshape(in_features,num_basis,out_features))    
        self.bias=nn.Parameter(torch.zeros(out_features,**factory_kwargs))
    
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
        if base_fn is ...:
            base_fn=torch.cos
        if base_fn is not None:
            self.base_fn=base_fn
            self.scale_base=torch.nn.Parameter(torch.ones(in_features,out_features,**factory_kwargs))
            torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
        else:
            self.base_fn=None
    
    def reset_parameters(self):
        out_features=self.out_features
        in_features=self.in_features
        num_basis=self.num_centers
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        weight=torch.zeros(out_features,in_features*num_basis,**factory_kwargs)
        # torch.nn.init.kaiming_uniform_(weight, a=2.963)
        gaussian_kaiming_uniform_(weight)
        
        self.weight.data=weight.reshape(in_features,num_basis,out_features)
        # self.bias.data=torch.zeros(out_features,**factory_kwargs)
    
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
        
        self.scale_base=torch.nn.Parameter(torch.ones(in_features,out_features,**factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
        
        
    def forward(self, x):
        batch_shape=x.shape[:-1]
        x=x.reshape(-1,self.in_features)
    
        base=self.base_fn(x).unsqueeze(-1)*self.scale_base.unsqueeze(0)
        
        y = self.rbf(x)  # [batch, in_features, num_centers]
    
        y=y.unsqueeze(-1) *self.weight.unsqueeze(0)
       
        # print(x.shape,self.in_features,self.out_features)
        y= y.sum(dim=(1,2))+self.bias+base.sum(dim=1)
        y=y.reshape(*batch_shape,self.out_features)
        return y