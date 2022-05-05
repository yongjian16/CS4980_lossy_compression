import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd, init
from torch.nn.modules.utils import _single, _pair, _triple

class FiConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=None, 
                 padding_mode='zeros', coefficient=0., error_type='uniform', batch_size=64):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.coefficient = coefficient
        self.error_type = error_type
        self.batch_size = batch_size
        super(FiConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        return ficonv2d().apply(input, weight, self.bias, self.stride, self.padding, 
                                self.dilation, self.groups, self.coefficient ,self.error_type, self.batch_size)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

class ficonv2d(torch.autograd.Function):
    
   @staticmethod
   def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, coefficient=0, error_type='uniform', batch_size=64):
     ctx.save_for_backward(input, weight)
     ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
     ctx.groups, ctx.coefficient = groups, coefficient
     ctx.error_type, ctx.batch_size = error_type, batch_size
     return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

   @staticmethod
   def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation
    groups, coefficient = ctx.groups, ctx.coefficient 
    error_type, batch_size = ctx.error_type, ctx.batch_size
    input +=  FI(input, weight, grad_output, coefficient, error_type, batch_size)
    x_grad = w_grad = None
    if ctx.needs_input_grad[0]:
      x_grad = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
    if ctx.needs_input_grad[1]:
      w_grad = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
    return x_grad, w_grad, None, None, None, None, None, None, None, None

def FI(activations, weight, grad, coefficient=0., error_type='uniform', batch_size=64):
    a_mean = activations.mean()
    sparsity = torch.mean((activations!=0).float()).cpu()
    error_bound = coefficient/np.sqrt(sparsity*batch_size)*a_mean.cpu().data
    if error_type == 'uniform':
        return 2*error_bound*(torch.rand(activations.shape)-0.5).cuda()
    if error_type == 'normal':
        return (error_bound/np.sqrt(3))*torch.randn(activations.shape).cuda()



