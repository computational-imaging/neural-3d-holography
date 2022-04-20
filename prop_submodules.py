"""
Modules for propagation

"""

import math
import torch
import torch.nn as nn
import utils
from unet import Conv2dSame


class Field2Input(nn.Module):
    """Gets complex-valued field and turns it into multi-channel images"""

    def __init__(self, input_res=(800, 1280), coord='rect', shared_cnn=False):
        super(Field2Input, self).__init__()
        self.input_res = input_res
        self.coord = coord.lower()
        self.shared_cnn = shared_cnn

    def forward(self, input_field):
        # If input field is slm phase
        if input_field.dtype == torch.float32:
            input_field = torch.exp(1j * input_field)

        input_field = utils.pad_image(input_field, self.input_res, pytorch=True, stacked_complex=False)
        input_field = utils.crop_image(input_field, self.input_res, pytorch=True, stacked_complex=False)

        # To use shared CNN, put everything into batch dimension;
        if self.shared_cnn:
            num_mb, num_dists = input_field.shape[0], input_field.shape[1]
            input_field = input_field.reshape(num_mb*num_dists, 1, *input_field.shape[2:])

        # Input format
        if self.coord == 'rect':
            stacked_input = torch.cat((input_field.real, input_field.imag), 1)
        elif self.coord == 'polar':
            stacked_input = torch.cat((input_field.abs(), input_field.angle()), 1)
        elif self.coord == 'amp':
            stacked_input = input_field.abs()
        elif 'both' in self.coord:
            stacked_input = torch.cat((input_field.abs(), input_field.angle(), input_field.real, input_field.imag), 1)

        return stacked_input


class Output2Field(nn.Module):
    """Gets complex-valued field and turns it into multi-channel images"""

    def __init__(self, output_res=(800, 1280), coord='rect', num_ch_output=1):
        super(Output2Field, self).__init__()
        self.output_res = output_res
        self.coord = coord.lower()
        self.num_ch_output = num_ch_output  # number of channels in output

    def forward(self, stacked_output):

        if self.coord in ('rect', 'both'):
            complex_valued_field = torch.view_as_complex(stacked_output.unsqueeze(4).
                                                         permute(0, 4, 2, 3, 1).contiguous())
        elif self.coord == 'polar':
            amp = stacked_output[:, 0:1, ...]
            phi = stacked_output[:, 1:2, ...]
            complex_valued_field = amp * torch.exp(1j * phi)
        elif self.coord == 'amp' or '1ch_output' in self.coord:
            complex_valued_field = stacked_output * torch.exp(1j * torch.zeros_like(stacked_output))

        output_field = utils.pad_image(complex_valued_field, self.output_res, pytorch=True, stacked_complex=False)
        output_field = utils.crop_image(output_field, self.output_res, pytorch=True, stacked_complex=False)

        if self.num_ch_output > 1:
            # reshape to original tensor shape
            output_field = output_field.reshape(output_field.shape[0] // self.num_ch_output, self.num_ch_output,
                                                *output_field.shape[2:])

        return output_field


class Conv2dField(nn.Module):
    """Apply 2d conv on amp or field"""

    def __init__(self, comp=False, conv_size=3):
        super(Conv2dField, self).__init__()
        self.comp = comp  # apply convolution on field
        self.conv_size = (conv_size, conv_size)
        if self.comp:
            self.conv_real = Conv2dSame(1, 1, conv_size)
            self.conv_imag = Conv2dSame(1, 1, conv_size)
            init_weight = torch.zeros(1, 1, *self.conv_size)
            init_weight[..., conv_size//2, conv_size//2] = 1.
            self.conv_real.net[1].weight = nn.Parameter(init_weight.detach().requires_grad_(True))
            self.conv_imag.net[1].weight = nn.Parameter(init_weight.detach().requires_grad_(True))
        else:
            self.conv = Conv2dSame(1, 1, conv_size, bias=False)
            init_weight = torch.zeros(1, 1, *self.conv_size)
            init_weight[..., conv_size//2, conv_size//2] = 1.
            self.conv.net[1].weight = nn.Parameter(init_weight.requires_grad_(True))

    def forward(self, input_field):

        # reshape tensor if number of channels > 1
        num_ch = input_field.shape[1]
        if num_ch > 1:
            batch_size = input_field.shape[0]
            input_field = input_field.reshape(batch_size * num_ch, 1, *input_field.shape[2:])

        if self.comp:
            # apply conv on complex fields
            real = self.conv_real(input_field.real) - self.conv_imag(input_field.imag)
            imag = self.conv_real(input_field.imag) + self.conv_imag(input_field.real)
            output_field = torch.view_as_complex(torch.stack((real, imag), -1))
        else:
            # apply conv on intensity
            output_amp = self.conv(input_field.abs()**2).abs().mean(dim=1, keepdims=True).sqrt()
            output_field = output_amp * torch.exp(1j * input_field.angle())

        # reshape to original tensor shape
        if num_ch > 1:
            output_field = output_field.reshape(batch_size, num_ch, *output_field.shape[2:])

        return output_field


class LatentCodedMLP(nn.Module):
    """
    concatenate latent codes in the middle of forward pass as well.
    put latent codes shape of (1, L, H, W) as a parameter for the forward pass.
    num_latent_codes: list of numbers of slices for each layer
    * so the sum of num_latent_codes should be total number of the latent codes channels
    """
    def __init__(self, num_layers=5, num_features=32, norm=None, num_latent_codes=None):
        super(LatentCodedMLP, self).__init__()

        if num_latent_codes is None:
            num_latent_codes = [0] * num_layers

        assert len(num_latent_codes) == num_layers

        self.num_latent_codes = num_latent_codes
        self.idxs = [sum(num_latent_codes[:y]) for y in range(num_layers + 1)]
        self.nets = nn.ModuleList([])
        num_features = [num_features] * num_layers
        num_features[0] = 1

        # define each layer
        for i in range(num_layers - 1):
            net = [nn.Conv2d(num_features[i] + num_latent_codes[i], num_features[i + 1], kernel_size=1)]
            if norm is not None:
                net += [norm(num_groups=4, num_channels=num_features[i + 1], affine=True)]
            net += [nn.LeakyReLU(0.2, True)]
            self.nets.append(nn.Sequential(*net))

        self.nets.append(nn.Conv2d(num_features[-1] + num_latent_codes[-1], 1, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.05)

    def forward(self, phases, latent_codes=None):

        after_relu = phases
        # concatenate latent codes at each layer and send through the convolutional layers
        for i in range(len(self.num_latent_codes)):
            if latent_codes is not None:
                after_relu = torch.cat((after_relu, latent_codes[:, self.idxs[i]:self.idxs[i + 1], ...]), 1)
            after_relu = self.nets[i](after_relu)

        # residual connection
        return phases - after_relu


class ContentDependentField(nn.Module):
    def __init__(self, num_layers=5, num_features=32, norm=nn.GroupNorm, latent_coords=False):
        """ Simple 5layers CNN modeling content dependent undiffracted light """

        super(ContentDependentField, self).__init__()

        if not latent_coords:
            first_ch = 1
        else:
            first_ch = 3

        net = [Conv2dSame(first_ch, num_features, kernel_size=3)]

        for i in range(num_layers - 2):
            if norm is not None:
                net += [norm(num_groups=2, num_channels=num_features, affine=True)]
            net += [nn.LeakyReLU(0.2, True),
                    Conv2dSame(num_features, num_features, kernel_size=3)]

        if norm is not None:
            net += [norm(num_groups=4, num_channels=num_features, affine=True)]

        net += [nn.LeakyReLU(0.2, True),
                Conv2dSame(num_features, 2, kernel_size=3)]

        self.net = nn.Sequential(*net)

    def forward(self, phases, latent_coords=None):
        if latent_coords is not None:
            input_cnn = torch.cat((phases, latent_coords), dim=1)
        else:
            input_cnn = phases

        return self.net(input_cnn)


class ProcessPhase(nn.Module):
    def __init__(self, num_layers=5, num_features=32, num_output_feat=0, norm=nn.BatchNorm2d, num_latent_codes=0):
        super(ProcessPhase, self).__init__()

        # avoid zero
        self.num_output_feat = max(num_output_feat, 1)
        self.num_latent_codes = num_latent_codes

        # a bunch of 1x1 conv layers, set by num_layers
        net = [nn.Conv2d(1 + num_latent_codes, num_features, kernel_size=1)]

        for i in range(num_layers - 2):
            if norm is not None:
                net += [norm(num_groups=2, num_channels=num_features, affine=True)]
            net += [nn.LeakyReLU(0.2, True),
                    nn.Conv2d(num_features, num_features, kernel_size=1)]

        if norm is not None:
            net += [norm(num_groups=2, num_channels=num_features, affine=True)]

        net += [nn.ReLU(True),
                nn.Conv2d(num_features, self.num_output_feat, kernel_size=1)]

        self.net = nn.Sequential(*net)

    def forward(self, phases):
        return phases - self.net(phases)


class SourceAmplitude(nn.Module):
    def __init__(self, num_gaussians=3, init_sigma=None, init_amp=0.7, x_s0=0.0, y_s0=0.0):
        super(SourceAmplitude, self).__init__()

        self.num_gaussians = num_gaussians

        if init_sigma is None:
            init_sigma = [100.] * self.num_gaussians  # default to 100 for all

        # create parameters for source amplitudes
        self.sigmas = nn.Parameter(torch.tensor(init_sigma))
        self.x_s = nn.Parameter(torch.ones(num_gaussians) * x_s0)
        self.y_s = nn.Parameter(torch.ones(num_gaussians) * y_s0)
        self.amplitudes = nn.Parameter(torch.ones(num_gaussians) / (num_gaussians) * init_amp)
        self.dc_term = nn.Parameter(torch.zeros(1))

        self.x_dim = None
        self.y_dim = None

    def forward(self, phases):
        # create DC term, then add the gaussians
        source_amp = torch.ones_like(phases) * self.dc_term
        for i in range(self.num_gaussians):
            source_amp += self.create_gaussian(phases.shape, i)

        return source_amp

    def create_gaussian(self, shape, idx):
        # create sampling grid if needed
        if self.x_dim is None or self.y_dim is None:
            self.x_dim = torch.linspace(-(shape[-1] - 1) / 2,
                                        (shape[-1] - 1) / 2,
                                        shape[-1], device=self.dc_term.device)
            self.y_dim = torch.linspace(-(shape[-2] - 1) / 2,
                                        (shape[-2] - 1) / 2,
                                        shape[-2], device=self.dc_term.device)

        if self.x_dim.device != self.sigmas.device:
            self.x_dim.to(self.sigmas.device).detach()
            self.x_dim.requires_grad = False
        if self.y_dim.device != self.sigmas.device:
            self.y_dim.to(self.sigmas.device).detach()
            self.y_dim.requires_grad = False

        # offset grid by coordinate and compute x and y gaussian components
        x_gaussian = torch.exp(-0.5 * torch.pow(torch.div(self.x_dim - self.x_s[idx], self.sigmas[idx]), 2))
        y_gaussian = torch.exp(-0.5 * torch.pow(torch.div(self.y_dim - self.y_s[idx], self.sigmas[idx]), 2))

        # outer product with amplitude scaling
        gaussian = torch.ger(self.amplitudes[idx] * y_gaussian, x_gaussian)

        return gaussian


def make_kernel_gaussian(sigma, kernel_size):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2
    variance = sigma**2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = ((1 / (2 * math.pi * variance))
                       * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)
                                   / (2 * variance)))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    return gaussian_kernel


def create_gaussian(shape, sigma=800, dev=torch.device('cuda')):
    # create sampling grid if needed
    shape_min = min(shape[-1], shape[-2])
    x_dim = torch.linspace(-(shape_min - 1) / 2,
                                    (shape_min - 1) / 2,
                                    shape[-1], device=dev)
    y_dim = torch.linspace(-(shape_min - 1) / 2,
                                    (shape_min - 1) / 2,
                                    shape[-2], device=dev)

    # offset grid by coordinate and compute x and y gaussian components
    x_gaussian = torch.exp(-0.5 * torch.pow(torch.div(x_dim, sigma), 2))
    y_gaussian = torch.exp(-0.5 * torch.pow(torch.div(y_dim, sigma), 2))

    # outer product with amplitude scaling
    gaussian = torch.ger(y_gaussian, x_gaussian)

    return gaussian




