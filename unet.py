import torch
import torch.nn as nn
from torch.nn import init
import functools


def norm_layer(norm_str):
    if norm_str.lower() == 'instance':
        return nn.InstanceNorm2d
    elif norm_str.lower() == 'group':
        return nn.GroupNorm
    elif norm_str.lower() == 'batch':
        return nn.BatchNorm2d


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 outer_skip=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.outer_skip = outer_skip
        if norm_layer == None:
            use_bias = True
        elif type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=5,
                             # Change kernel size changed to 5 from 4 and padding size from 1 to 2
                             stride=2, padding=2, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        if norm_layer is not None:
            if norm_layer == nn.GroupNorm:
                downnorm = norm_layer(8, inner_nc)
            else:
                downnorm = norm_layer(inner_nc)
        else:
            downnorm = None
        uprelu = nn.ReLU(True)
        if norm_layer is not None:
            if norm_layer == nn.GroupNorm:
                upnorm = norm_layer(8, outer_nc)
            else:
                upnorm = norm_layer(outer_nc)
        else:
            upnorm = None

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downrelu]
            up = [upconv]  # Removed tanh and uprelu
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if norm_layer is not None:
                down = [downconv, downnorm, downrelu]
                up = [upconv, upnorm, uprelu]
            else:
                down = [downconv, downrelu]
                up = [upconv, uprelu]

            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if norm_layer is not None:
                down = [downconv, downnorm, downrelu]
                up = [upconv, upnorm, uprelu]
            else:
                down = [downconv, downrelu]
                up = [upconv, uprelu]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost and not self.outer_skip:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


def init_latent(latent_num, wavefront_res, ones=False):
    if latent_num > 0:
        if ones:
            latent = nn.Parameter(torch.ones(1, latent_num, *wavefront_res,
                                             requires_grad=True))
        else:
            latent = nn.Parameter(torch.zeros(1, latent_num, *wavefront_res,
                                              requires_grad=True))
    else:
        latent = None
    return latent


def apply_net(net, input, latent_code, complex=False):
    if net is None:
        return input
    if complex: # Only valid for single batch or single channel complex inputs and outputs
        multi_channel = (input.shape[1] > 1)
        if multi_channel:
            input = torch.view_as_real(input[0,...])
        else:
            input = torch.view_as_real(input[:,0,...])
        input = input.permute(0,3,1,2)
    if latent_code is not None:
        input = torch.cat((input, latent_code), dim=1)
    output = net(input)
    if complex:
        if multi_channel:
            output = output.permute(0,2,3,1).unsqueeze(0)
        else:
            output = output.permute(0,2,3,1).unsqueeze(1)
        output = torch.complex(output[...,0], output[...,1])
    return output


def init_weights(net, init_type='normal', init_gain=0.02, outer_skip=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=1, output_nc=1, num_downs=8, nf0=32, max_channels=512,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False, outer_skip=True,
                 half_channels=False, eighth_channels=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.outer_skip = outer_skip
        self.input_nc = input_nc

        if eighth_channels:
            divisor = 8
        elif half_channels:
            divisor = 2
        else:
            divisor = 1
            # construct unet structure

        assert num_downs >= 2

        # Add the innermost layer
        unet_block = UnetSkipConnectionBlock(min(2 ** (num_downs - 1) * nf0, max_channels) // divisor,
                                             min(2 ** (num_downs - 1) * nf0, max_channels) // divisor,
                                             input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)

        for i in list(range(1, num_downs - 1))[::-1]:
            if i == 1:
                norm = None  # Praneeth's modification
            else:
                norm = norm_layer

            unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels) // divisor,
                                                 min(2 ** (i + 1) * nf0, max_channels) // divisor,
                                                 input_nc=None, submodule=unet_block,
                                                 norm_layer=norm,
                                                 use_dropout=use_dropout)

        # Add the outermost layer
        self.model = UnetSkipConnectionBlock(min(nf0, max_channels) // divisor,
                                             min(2 * nf0, max_channels) // divisor,
                                             input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=None, outer_skip=self.outer_skip)
        if self.outer_skip:
            self.additional_conv = nn.Conv2d(input_nc + min(nf0, max_channels) // divisor, output_nc,
                                             kernel_size=4, stride=1, padding=2, bias=True)
        else:
            self.additional_conv = nn.Conv2d(min(nf0, max_channels) // divisor, output_nc,
                                             kernel_size=4, stride=1, padding=2, bias=True)

    def forward(self, cnn_input):
        """Standard forward"""
        output = self.model(cnn_input)
        output = self.additional_conv(output)
        output = output[:,:,:-1,:-1]
        return output


class Conv2dSame(torch.nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)
