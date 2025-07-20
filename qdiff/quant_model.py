import logging
import torch
import torch.nn as nn
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock, QuantHunyuanBlock
from qdiff.quant_block import QuantDiffBTB, QuantDiffRB
from qdiff.quant_layer import QuantModule, StraightThrough
from ldm.modules.attention import BasicTransformerBlock
import random
import numpy as np

logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])

        # Hacked changes for SDXL
        if hasattr(model, "dtype"):
            self.dtype=model.dtype
        if hasattr(model, "config"):
            self.config = model.config
            if hasattr(model, "add_embedding"):
                #self.add_embedding = model.add_embedding
                #self.in_features = model.in_features
                self.forward = self.forward_diffusers
            else:
                self.forward = model.forward

        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)
        self.set_attn_weight_mantissa_bits(weight_quant_params)
        self.set_ff1_weight_mantissa_bits(weight_quant_params)
        self.set_asym_for_sm(act_quant_params)
        

    # Only support for QuantDiffBTB softmax
    # Change the sign_bits to 0, mantissa remain unchange
    # Hence exponent bit plus one
    def set_asym_for_sm(self, act_quant_params: dict = {}):
        if act_quant_params['asym_softmax'] == True:
            logger.info('Using Symmetric Quantization for Softmax')
            return
        logger.info('Using Asymmetric Quantization for Softmax')
        for m in self.model.modules():
            if isinstance(m, QuantDiffBTB):
                m.attn1.act_quantizer_w.sign_bits = 0
                if m.attn2 is not None:
                    m.attn2.act_quantizer_w.sign_bits = 0
    
    def set_attn_weight_mantissa_bits(self, weight_quant_params: dict = {}):
        if weight_quant_params['attn_weight_mantissa'] is None:
            logger.info('No specific attention M')
            return
        num = weight_quant_params['attn_weight_mantissa']
        logger.info(f'Change M to {num} for PixArt Attention')
        for m in self.model.modules():
            if isinstance(m, QuantDiffBTB):
                m.attn1.to_q.weight_quantizer.mantissa_bits[0] = num
                m.attn1.to_k.weight_quantizer.mantissa_bits[0] = num
                m.attn1.to_v.weight_quantizer.mantissa_bits[0] = num
                m.attn1.to_out[0].weight_quantizer.mantissa_bits[0] = num
                if m.attn2 is not None:
                    m.attn2.to_q.weight_quantizer.mantissa_bits[0] = num
                    m.attn2.to_k.weight_quantizer.mantissa_bits[0] = num
                    m.attn2.to_v.weight_quantizer.mantissa_bits[0] = num
                    m.attn2.to_out[0].weight_quantizer.mantissa_bits[0] = num
    
    # Just the first layer (before the GELU)
    def set_ff1_weight_mantissa_bits(self, weight_quant_params: dict = {}):
        if weight_quant_params['ff_weight_mantissa'] is None:
            logger.info('No specific Feed Forward M')
            return
        num = weight_quant_params['ff_weight_mantissa']
        logger.info(f'Change M to {num} for the first layer in PixArt FeedForward')
        for m in self.model.modules():
            if isinstance(m, QuantDiffBTB):
                m.ff.net[0].proj.weight_quantizer.mantissa_bits[0] = num
                #m.ff.net[2].weight_quantizer.mantissa_bits[0] = num


    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            #print(name)
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # and name not in ['out.0', 'out.1', 'out.2']: # nn.Conv1d
            #if isinstance(child_module, (nn.Conv1d, nn.Linear)):  # Quantizing only the attention.
            #if isinstance(child_module, nn.Conv2d): # or (isinstance(child_module, (nn.Linear, nn.Conv1d)) and "emb" in child_module.nametag):
                #print("Quantizing ", name)
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock, QuantDiffBTB, QuantHunyuanBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        #print("Forward called")
        return self.model(x, timesteps, context)
    
    def forward_diffusers(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict=False,
                ):
        #print("Forward called")
        return self.model(
            sample=latent_model_input, 
            timestep=t, 
            encoder_hidden_states=encoder_hidden_states, 
            cross_attention_kwargs=cross_attention_kwargs, 
            added_cond_kwargs=added_cond_kwargs, 
            return_dict=return_dict)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            elif isinstance(m, QuantDiffBTB):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_q.running_stat = running_stat
                        m.attn2.act_quantizer_k.running_stat = running_stat
                        m.attn2.act_quantizer_v.running_stat = running_stat
                        m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantDiffBTB, QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt

class QuantModelSelect(nn.Module):

    def __init__(self, model: nn.Module, target_module: str, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])

        #for name, child_module in model.named_modules():
        #    print(name)

        # Hacked changes for SDXL
        self.dtype=model.dtype
        if hasattr(model, "config"):
            self.config = model.config
            self.forward = self.forward_diffusers
        if hasattr(model, "add_embedding"):
            self.add_embedding = model.add_embedding
            #self.in_features = model.in_features


        self.quant_module_refactor(self.model, target_module, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, target_module: str, weight_quant_params: dict = {}, act_quant_params: dict = {}, cummulative_name=""):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            #print(name)
            if cummulative_name == "":
                module_full_name = name
            else:
                module_full_name = ".".join([cummulative_name, name])
            #print(module_full_name)
            if module_full_name.startswith(target_module) and isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                #print(f"Firing on {module_full_name}")
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, target_module, weight_quant_params, act_quant_params, cummulative_name=module_full_name)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params, )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)

    def forward_diffusers(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict=False,
                ):
        return self.model(
            sample=latent_model_input, 
            timestep=t, 
            encoder_hidden_states=encoder_hidden_states, 
            cross_attention_kwargs=cross_attention_kwargs, 
            added_cond_kwargs=added_cond_kwargs, 
            return_dict=return_dict)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            elif isinstance(m, QuantDiffBTB):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_q.running_stat = running_stat
                        m.attn2.act_quantizer_k.running_stat = running_stat
                        m.attn2.act_quantizer_v.running_stat = running_stat
                        m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantDiffBTB, QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt


class QuantModelMultiQ(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, prefix=None, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])

        #for name, child_module in model.named_modules():
        #    print(name)

        # Hacked changes for SDXL
        if hasattr(model, "dtype"):
            self.dtype=model.dtype
        if hasattr(model, "config"):
            self.config = model.config
            if hasattr(model, "add_embedding"):
                self.add_embedding = model.add_embedding
                #self.in_features = model.in_features
                self.forward = self.forward_diffusers
            else:
                self.forward = model.forward

        # For the skip-connect splits in SDv1.5, which is defined in this repo.
        # For SDXL it is always enabled.
        if hasattr(self.model, "split"):
            self.model.split = True

        # HunYuan-DiT
        if hasattr(self.model, "inner_dim"):
            self.inner_dim = self.model.inner_dim
        if hasattr(self.model, "num_heads"):
            self.num_heads = self.model.num_heads

        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, prefix=prefix)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, prefix=None):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            #print(name)
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # and name not in ['out.0', 'out.1', 'out.2']: # nn.Conv1d
            #if isinstance(child_module, (nn.Conv1d, nn.Linear)):  # Quantizing only the attention.
            #if isinstance(child_module, nn.Conv2d): # or (isinstance(child_module, (nn.Linear, nn.Conv1d)) and "emb" in child_module.nametag):
                #print("Quantizing ", name)
                # NOTE REVERSE THIS MESS AFTER SDXL!
                if prefix is not None:
                    if prefix in child_module.nametag:
                        setattr(module, name, QuantModuleMulti(
                        child_module, weight_quant_params, act_quant_params))
                        prev_quantmodule = getattr(module, name)
                    else:
                        continue
                else:
                    setattr(module, name, QuantModuleMulti(
                        child_module, weight_quant_params, act_quant_params))
                    prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, prefix=prefix)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params, )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModuleMulti, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        #print("Forward called")
        return self.model(x, timesteps, context)
    
    def forward_diffusers(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict=False,
                ):
        #print("Forward called")
        return self.model(
            sample=latent_model_input, 
            timestep=t, 
            encoder_hidden_states=encoder_hidden_states, 
            cross_attention_kwargs=cross_attention_kwargs, 
            added_cond_kwargs=added_cond_kwargs, 
            return_dict=return_dict)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            elif isinstance(m, QuantDiffBTB):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_q.running_stat = running_stat
                        m.attn2.act_quantizer_k.running_stat = running_stat
                        m.attn2.act_quantizer_v.running_stat = running_stat
                        m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantDiffBTB, QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt


    # TODO functions that we want
    # 1. Goes through model. Generate dict. DONE
    #   Key is nametag
    #   Value is MultiQuantizer
    # 2. Loading function (REMEMBER WHAT WE SAID ABOUT PASSING A TENSOR THROUGH ON SDXL AND SDV1.5)
    # 3. Quantify savings refactor DONE
    # 4. Get config refactor DONE
    # 5. Randomly assign setting 
    # NOTE: I make this all the time.
    # Considered creating it with the QuantModelMultiQ and keeping it for faster speed
    # But went against that due to the weight_quantizer_0 modules for SDXL and SDv1.5
    def get_all_multiquantizers(self, module=None):
        mq_dict = {}
        if module is None:
            module = self.model
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            if isinstance(child_module, MultiQuantizer):
                mq_dict[child_module.nametag] = child_module
            else:
                mq_dict = {**mq_dict, **self.get_all_multiquantizers(child_module)}
        return mq_dict

    def get_all_quantmodules(self, module=None):
        qm_dict = {}
        if module is None:
            module = self.model
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            if isinstance(child_module, QuantModuleMulti):
                qm_dict[child_module.nametag] = child_module
            else:
                qm_dict = {**qm_dict, **self.get_all_quantmodules(child_module)}
        return qm_dict

    def quantify_model_savings(self):
        original_size, quant_size = 0, 0        
        for mq_obj in self.get_all_multiquantizers().values():
            module_og_size, module_q_size = mq_obj.quantify_sizes()
            original_size += module_og_size
            quant_size += module_q_size

        print("Original size in bytes: ", original_size)
        print("Quantized size in bytes: ", quant_size)
        print(f"Reduction to {(quant_size/original_size)*100}% of FP model size.")
        return quant_size

    def get_current_quant_config(self, skip_act=True, include_stats=False):
        mq_config = {}
        for mq_name, mq_obj in self.get_all_multiquantizers().items():
            if skip_act and "act_quantizer" in mq_name:
                continue
            mq_config[mq_name] = [mq_obj.quant_method]
            if include_stats:
                module_og_size, module_q_size = mq_obj.quantify_sizes()
                module_error = mq_obj.loss_dict[mq_obj.quant_method].item()
                mq_config[mq_name] += [module_og_size, module_q_size, module_error]
        return mq_config

    """
    def adhoc_size_constraint(self, sdv15=False):
        multiplier = 32 if sdv15 else 16
        quant_size = self.quantify_model_savings()
        ave_bits = quant_size * multiplier
        if ave_bits < 3.4:
            return True
        elif ave_bits > 3.6 and ave_bits < 4.0:
            return True
        return
    """

    def set_random_config(self, skip_act=True, exclude_quantile=True, dit=False, sdv15=False, sdxl=False, exclude_2=True, a_l=0.0, a_u=1.0):  # I'm just going to set this to true. We 
        mq_dict = self.get_all_multiquantizers()
        ave_bits_p = np.random.uniform(low=a_l, high=a_u)
        bits = np.random.choice([3, 4], size=len(mq_dict), p=[ave_bits_p, 1-ave_bits_p])
        i = 0
        for mq_name, mq_obj in mq_dict.items():
            if skip_act and "act_quantizer" in mq_name:
                continue
            options = [o for o in mq_obj.loss_dict.keys()]
            if exclude_2:
                options = [o for o in options if "-2" not in o]
            if exclude_quantile:
                options = [o for o in options if "quantile" not in o]
            config = random.choice(options)
            config = config[:-1]
            config += str(bits[i])
            #mq_obj.set_quant_method(random.choice(options))
            mq_obj.set_quant_method(config)
            i += 1
        # This conditional is thrown in because the DiT design is weird in that all transformer blocks use THE SAME timestep embedding Linear-SiLU-Linear sequential. 
        # So, in code, we will generate quantizers for all of them, even though its referring to the same weights actually. Makes the structure very weird.
        # Basically, when randomly sampling, since this is supposed to be 1 set of weights, we assign the same quant_method across the board.
        if dit:
            for i in range(1, 28):
                mq_dict[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].set_quant_method(mq_dict['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].quant_method)
                mq_dict[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].set_quant_method(mq_dict['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].quant_method)


    def load_mq_state_dicts(self, filepath=None, sd=None):
        if filepath is None:
            assert sd is not None
        elif sd is None:
            assert filepath is not None
            sd = torch.load(filepath)
        qm_dict = self.get_all_quantmodules()
        for qm_name in qm_dict.keys():
            sub_dict = {sd_key: sd[sd_key] for sd_key in sd.keys() if sd_key.startswith(qm_name)}
            qm_dict[qm_name].custom_sd_load(sub_dict)
            for sd_key in sub_dict.keys():
                del sd[sd_key]
        return sd
    
    def load_quant_config(self, filepath):
        import pickle
        with open(filepath, "rb") as f:
            quant_config_dict = pickle.load(f)

        all_wqs = self.get_all_multiquantizers()
        for k, v in quant_config_dict.items():
            all_wqs[k].set_quant_method(v)

        if "dit" in filepath:
            for i in range(1, 28):
                all_wqs[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].set_quant_method(all_wqs['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].quant_method)
                all_wqs[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].set_quant_method(all_wqs['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].quant_method)