import torch
from torch import nn
import logging
from qdiff.quant_layer import UniformAffineQuantizer, round_ste
from qdiff.quantizers.fp8_quantizer import FPQuantizer, quantize_to_fp8_ste_MM, quantize_to_fp8_ste_MM_soft_targets, quantize_to_fp8_rest_scale

logger = logging.getLogger(__name__)


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.fp = uaq.fp
        self.mantissa_bits = uaq.mantissa_bits
        self.sign_bits = 1
        self.maxval = uaq.delta
        self.fp_biased_adaround = uaq.fp_biased_adaround

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False
        self.scale = None

        # NOTE for input/output channel wise weight quantization
        # 0 for output channel wise, 1 for input channel wise
        # Input tensor x would be (output_channels, input_channels, ..., ...)
        # Just transpose it as (input_channels, output_channels, ..., ...)
        # And reverse back when return
        self.dim = uaq.dim

        # NOTE for group quantization
        # indicate by a flag self.group_quant
        # Group number = 64 (optional choice = 48, 128)
        # For each channel
        self.group_quant = uaq.group_quant
        self.group_size = uaq.group_size

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())
        #self.max_delta = None
        #if hasattr(uaq, "max_zero_point"):
        #    self.max_delta = uaq.max_delta
        #print("Adaround!!!!!!") # NOTE OKAY, it works LOL

    """
    def report_delta_shift(self):
        if self.max_delta is None:
            print("No delta reported")
            return
        percent_change = (self.delta - self.max_delta) / self.max_delta
        percent_change *= 100
        p_mu, p_sig, p_min, p_max = torch.mean(percent_change), torch.std(percent_change), percent_change.min().item(), percent_change.max().item()
        return f"{p_min}, {p_max}, {p_mu}, {p_sig}"
    """
        
    def forward(self, x):
        # NOTE for input/output channel
        #if self.dim == 1:
        #   x = torch.transpose(x, 0, 1)

        # NOTE for group quantization
        # applying independent quantization to groups of g consecutive weights
        if self.group_quant == True:
            x_old = x
            x = x.view(-1, self.group_size)

        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            logger.info('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            if self.fp == False:
                x_floor = torch.floor(x / self.delta)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
            else:
                if self.soft_targets:
                    x_dequant = quantize_to_fp8_ste_MM_soft_targets(
                    x, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits, self.get_soft_targets()      
                    )
                else:
                    x_dequant = quantize_to_fp8_ste_MM_soft_targets(
                    x, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits, (self.alpha >= 0).float()      
                    )

                # NOTE for group quantization
                if self.group_quant == True:
                    x_dequant = x_dequant.view(x_old.shape)
                return x_dequant
        else:
            raise ValueError('Wrong rounding mode')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q

    def get_soft_targets(self):
        # NOTE this if statement is entirely for FP biased Adaround
        # If not using FP biased Adaround, remove the whole if
        # So I add a flag self.fp_biased_adaround here, True then do FP biased adaround
        if (self.fp == True) and (self.fp_biased_adaround == True):
            return torch.clamp(torch.sigmoid(self.alpha / self.scale) * (self.zeta - self.gamma) + self.gamma, 0, 1)
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        # NOTE for group quantization
        # applying independent quantization to groups of g consecutive weights
        if self.group_quant == True:
            x_old = x
            x = x.view(-1, self.group_size)
        if self.fp == True:
            rest, self.scale = quantize_to_fp8_rest_scale(x, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
            if (self.fp_biased_adaround == True):
                # If self.fp_biased_adaround == False, equal to remove this line
                alpha = self.scale * alpha # NOTE FP biased Adaround (if not using FP biased, just remove this line)
            self.alpha = nn.Parameter(alpha)
            return
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            # logger.info('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def extra_repr(self):
        s = 'bit={n_bits}, symmetric={sym}, round_mode={round_mode}, M={mantissa_bits}' 
        return s.format(**self.__dict__)
