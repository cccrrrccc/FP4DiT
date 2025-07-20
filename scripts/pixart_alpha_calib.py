from diffusers import PixArtAlphaPipeline
#from sdxl_wrapper import StableDiffusionXLPipeline, 
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from qdiff import QuantModel
import sys
from copy import deepcopy
from pytorch_lightning import seed_everything

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
from qdiff import QuantModel

# TODO pull actual images, and prompts, from COCO.
# TODO change location
image_sample = "/home/ruichen/data/coco/val2017/000000462614.jpg"

timesteps = list(reversed(range(50)))

#quant_params = [int(sys.argv[1]), int(sys.argv[2])]
#print(quant_params)
seed_everything(42)

if __name__ == "__main__":

    pipeline = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16).to("cuda")
    #pipeline.upcast_vae() # NOTE: This was necessary for SDXL
    #pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    vae = pipeline.vae
    noise_scheduler = pipeline.scheduler
    mse = torch.nn.MSELoss()

    #wq_params = {'n_bits': quant_params[0], 'channel_wise': True, 'scale_method': 'max'}
    #aq_params = {'n_bits': quant_params[1], 'channel_wise': False, 'scale_method': 'max', 'leaf_param':  True if quant_params[1] < 10 else False}

    # TODO code for quantizing U-Net for certain steps.
    #transformer = qnn # pipeline.transformer

    # Preprocessing the datasets.
    coco_transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512) if False else transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip() if False else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # NOTE We're using val for now but should use train.
    coco_ds = CocoDetection(root="/datasets/coco/train2017/", annFile="/datasets/coco/annotations/captions_train2017.json", transform=coco_transform)

    # NOTE this controlled #images, for debugging purposes
    #num_images, batch_size = 1024, 1
    num_images, batch_size = 128, 1
    coco_ds.ids = coco_ds.ids[-num_images:] # To ensure some difference...

    coco_dl = DataLoader(coco_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    # NOTE controls timesteps
    desired_timesteps = 20
    timestep_list = list(range(desired_timesteps))
    timestep_list.reverse()

    xs, time_calib, all_pe, all_ppe, all_tids = torch.randn(desired_timesteps, num_images, 4, 64, 64), torch.zeros(desired_timesteps, num_images).long(), [], torch.randn(num_images, 1280), torch.randn(1, 6)
    all_uncond_pe = []
    assert(all_uncond_pe == all_pe)
    with torch.no_grad():

        # Upon Negar's suggestion, computing this early.
        latents_list = []
        for img_info in coco_dl:
            image_sample = torch.cat([img[0].unsqueeze(0) for img in img_info], axis=0)
            latents_list.append(vae.encode(image_sample.to('cuda:0', dtype=torch.float16)).latent_dist.sample().to(torch.float16) * vae.config.scaling_factor)
        latents = torch.cat(latents_list, dim=0)
        #print(latents)
        noise = torch.randn_like(latents)
        #print(noise.shape)

        for ts in timestep_list:
            os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
            loss = 0
            #qnn = QuantModel(deepcopy(pipeline.transformer), weight_quant_params=wq_params, act_quant_params=aq_params, act_quant_mode="qdiff", sm_abit=16)
            #qnn.cuda()
            #qnn.eval()
            #qnn.set_quant_state(weight_quant=True if quant_params[0] < 10 else False,
            #            act_quant=True if quant_params[1] < 10 else False)
            #transformer = qnn
            pe_list = []
            uncond_pe_list = []
            for i, img_info in enumerate(coco_dl):
                #print("======")
                image_sample = torch.cat([img[0].unsqueeze(0) for img in img_info], axis=0)
                prompt_list = [p[1][0]['caption'] for p in img_info]

                timesteps = torch.Tensor([ts] * len(prompt_list)) #.long()
                timesteps = timesteps.cuda()

                # TODO the text here is slightly different for SDXL
                #print(latents)
                noisy_latents = noise_scheduler.add_noise(latents[i*batch_size:(i+1)*batch_size, :, :, :], 
                                                          noise[i*batch_size:(i+1)*batch_size, :, :, :],
                                                          timesteps)
                noisy_latents = noise_scheduler.scale_model_input(noisy_latents, ts)
                
                #noisy_latents = pipeline.scheduler.step(noise[i*batch_size:(i+1)#*batch_size, :, :, :], timesteps, latents[i*batch_size:(i+1)#*batch_size, :, :, :], return_dict=False)[0]

                # TODO actual COCO prompt
                prompt_embeds, _, negative_prompt_embeds, _ = pipeline.encode_prompt(prompt_list) #,
                                                                #torch.device('cuda:0'),
                                                                #num_images_per_prompt=1,
                                                                #do_classifier_free_guidance=False) #[0]
                #encoder_hidden_states = encoder_hidden_states[:len(prompt_list), :, :]
                #prompt_tokens = pipeline.tokenizer(prompt_list)
                #encoder_hidden_states = pipeline.text_encoder(prompt_tokens, return_dict=False)[0]
                #print("EHS:", encoder_hidden_states.shape)
                #print(negative_prompt_embeds.shape)
                #print(prompt_embeds.shape)
                print(prompt_list)
                #assert(negative_prompt_embeds.shape != prompt_embeds.shape)

                #height = pipeline.default_sample_size * pipeline.vae_scale_factor
                #width = pipeline.default_sample_size * pipeline.vae_scale_factor

                #original_size = (height, width)
                #target_size = (height, width)

                #add_time_ids = pipeline._get_add_time_ids(
                #    original_size, (0, 0), target_size, dtype=prompt_embeds.dtype
                #)
                #add_time_ids = add_time_ids.to("cuda").repeat(batch_size * 1, 1)
                #added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}  # SDXL
                added_cond_kwargs = {'resolution': None, 'aspect_ratio': None}  # PixArt
                #print(pooled_prompt_embeds.shape)
                #print(add_time_ids.shape)
                #print(noisy_latents.shape)
                #print(timesteps.shape)
                #print(prompt_embeds.shape)
                #model_pred = pipeline.transformer(hidden_states=noisy_latents, timestep=timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs, #cross_attention_kwargs={},
                                #return_dict=False
                #                )[0]
                # NOTE: 8, 64, 64 -> First 4 channels is for CS, last 4 are for UCS.
                # NOTE: What this means is that the tuples are not (xs, ts, cs), (xs, ts, ucs), but (xs, ts, cs, ucs)
                #print(model_pred.shape)
                
                #added_cond_kwargs['text_embeds'] = added_cond_kwargs['text_embeds'].detach().cpu()
                #added_cond_kwargs['time_ids'] = added_cond_kwargs['time_ids'].detach().cpu()
                #aka.append(added_cond_kwargs)
                pe_list.append(prompt_embeds.detach().cpu())
                uncond_pe_list.append(negative_prompt_embeds.detach().cpu())
                time_calib[ts, i] = ts
                xs[ts, i, :, :, :] = noisy_latents.detach().cpu() #model_pred.detach().cpu()
                #all_ppe[i, :] = added_cond_kwargs['text_embeds']
                
            #del transformer
            #del qnn
            torch.cuda.empty_cache()
            all_pe.append(torch.cat(pe_list, dim=0))
            all_uncond_pe.append(torch.cat(uncond_pe_list, dim=0))
        all_pe = torch.cat([x.unsqueeze(0) for x in all_pe], dim=0)
        all_uncond_pe = torch.cat([x.unsqueeze(0) for x in all_uncond_pe], dim=0)
        #all_tids = added_cond_kwargs['time_ids']
        # NOTE Need to add UCS support

        print(all_pe.shape)
        print(all_uncond_pe.shape)
        assert(all_uncond_pe.shape == all_pe.shape)

        assert(torch.equal(all_uncond_pe[0][0], all_uncond_pe[0][1])) # prompt("") should remain unchange across different image
        assert(not torch.equal(all_pe[0][0], all_pe[0][1])) # prompt(different prompt) should be different across different image

        torch.save({'xs': xs, 'ts': time_calib, 'cs': all_pe, 'ucs': all_uncond_pe}, #, 'text_embeds': all_ppe, 'time_ids': all_tids},
                    "pixart_calib_brecq.pt")
