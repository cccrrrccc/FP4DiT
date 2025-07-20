import os
import torch

PIXART_TXT_FILE_LOC = os.sep.join(["captions", "pixart_samples.txt"])
COCO_VAL_CAPTIONS = os.sep.join(["captions", "captions_val2017.json"])
COCO_2014_CAPTIONS = os.sep.join(["captions", "annots_10k.txt"])
HPSV2_CAPTIONS = os.sep.join(["captions", "hpsv2.txt"])

def get_captions(name, model, coco_9k=False, coco_10k=False, coco2014=False, pixart=False, hpsv2=False):
    if hpsv2:
        assert not pixart
        assert not coco_9k
        assert not coco_10k
        tag = "_".join([name, "hpsv2"])
    elif pixart:
        assert not coco_9k
        assert not coco_10k
        tag = "_".join([name, "pixart"])
    elif coco2014:
        tag = "_".join([name, "coco_2014"])
    elif coco_9k:
        assert not coco_10k
        assert not pixart
        tag = "_".join([name, "coco_10k"])
    elif coco_10k:
        assert not coco_9k
        assert not pixart
        tag = "_".join([name, "coco_10k"])
    else:
        tag = "_".join([name, "coco_10k"])
    
    # Check if precomputed prompt embedding file already exists
    embedding_fname = os.sep.join(["captions", tag+".pt"])
    print("Checking for file ", embedding_fname)
    if os.path.isfile(embedding_fname):
        print("File found! Loading precomputed prompt embeddings")
        embedding_dict = torch.load(embedding_fname, map_location="cpu")
        prompt_embeds = embedding_dict['prompt_embeds']
        prompt_attention_masks = embedding_dict['prompt_attention_masks']
        if coco_9k:
            prompt_embeds = prompt_embeds[1000:]
            prompt_attention_masks = prompt_attention_masks[1000:]
        elif not pixart and not coco_10k and not coco2014 and not hpsv2: # coco_1k
            prompt_embeds = prompt_embeds[:1000]
            prompt_attention_masks = prompt_attention_masks[:1000]
        negative_prompt_embeds = embedding_dict['negative_prompt_embeds'].to('cuda')
        negative_prompt_attention_mask = embedding_dict['negative_prompt_attention_mask'].to('cuda')
        return prompt_embeds, prompt_attention_masks, negative_prompt_embeds, negative_prompt_attention_mask
    else:
        print("File not found. Precomputing prompt embeddings...")
        if hpsv2:
            with open("/home/kgmills/coco_bs/hpsv2_prompts.txt", 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        elif pixart:
            with open(PIXART_TXT_FILE_LOC, 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        elif coco2014:
            import pickle
            with open("/home/kgmills/coco_bs/annots_10k.pkl", "rb") as f:
                captions = pickle.load(f)
            prompt_list = [c['caption'] for c in captions]
        else:
            import json
            with open(COCO_VAL_CAPTIONS) as f:
                captions = json.load(f)
            captions_list = captions['annotations'][:10000]
            """
            if coco_10k:
                captions_list = captions_list[:10000]
            elif coco_9k:
                captions_list = captions_list[1000:10000]
            else:
                captions_list = captions_list[:1000]
            """
            prompt_list = [c['caption'] for c in captions_list]
            
        pes, pams, npe, npam = [], [], None, None
        with torch.no_grad():
            for i, prompt in enumerate(prompt_list):
                print(i, prompt)
                if i == 0:
                    pe, pam, npe, npam = model.encode_prompt(prompt=prompt)
                else:
                    pe, pam, _, _ = model.encode_prompt(prompt=prompt)
                    
                pes.append(pe.to('cpu'))
                pams.append(pam.to('cpu'))
            pes = torch.cat(pes, dim=0)
            pams = torch.cat(pams, dim=0)
        embedding_dict = {
            'prompt_embeds': pes,
            'prompt_attention_masks': pams,
            'negative_prompt_embeds': npe,
            'negative_prompt_attention_mask': npam
        }
        print("Saving precomputed prompt embeddings to disk: ", embedding_fname)
        torch.save(embedding_dict, embedding_fname)
        print("Save successful!")
        npe = npe.to("cuda")
        npam = npam.to("cuda")
        if coco_9k:
            return pes[1000:], pams[1000:], npe, npam
        elif not pixart and not coco_10k and not coco2014 and not hpsv2:
            return pes[:1000], pams[:1000], npe, npam
        else:
            return pes, pams, npe, npam
        
    
def get_captions_hunyuan(name, model, coco_9k=False, coco_10k=False, coco2014=False, pixart=False, hpsv2=False):
    assert name == "hunyuan"
    if hpsv2:
        assert not pixart
        assert not coco_9k
        assert not coco_10k
        tag = "_".join([name, "hpsv2"])
    elif pixart:
        assert not coco_9k
        assert not coco_10k
        tag = "_".join([name, "pixart"])
    elif coco2014:
        tag = "_".join([name, "coco_2014"])
    elif coco_9k:
        assert not coco_10k
        assert not pixart
        tag = "_".join([name, "coco_10k"])
    elif coco_10k:
        assert not coco_9k
        assert not pixart
        tag = "_".join([name, "coco_10k"])
    elif hpsv2:
        assert not coco_9k
        assert not pixart
        assert not coco_10k
        assert not coco2014
        tag = "_".join([name, "hpsv2"])
    else:
        tag = "_".join([name, "coco_10k"])
    
    # Check if precomputed prompt embedding file already exists
    embedding_fname = os.sep.join(["captions", tag+".pt"])
    print("Checking for file ", embedding_fname)
    if os.path.isfile(embedding_fname):
        print("File found! Loading precomputed prompt embeddings")
        embedding_dict = torch.load(embedding_fname, map_location="cpu")
        prompt_embeds = embedding_dict['prompt_embeds']
        prompt_embeds_2 = embedding_dict['prompt_embeds_2']
        prompt_attention_masks = embedding_dict['prompt_attention_masks']
        prompt_attention_masks_2 = embedding_dict['prompt_attention_masks_2']
        if coco_9k:
            prompt_embeds = prompt_embeds[1000:]
            prompt_embeds_2 = prompt_embeds_2[1000:]
            prompt_attention_masks = prompt_attention_masks[1000:]
            prompt_attention_masks_2 = prompt_attention_masks_2[1000:]
        elif coco2014:
            prompt_embeds = prompt_embeds[:5000]
            prompt_embeds_2 = prompt_embeds_2[:5000]
            prompt_attention_masks = prompt_attention_masks[:5000]
            prompt_attention_masks_2 = prompt_attention_masks_2[:5000]
        #elif hpsv2: # resume from image 2313
        #    prompt_embeds = prompt_embeds[2313:]
        #    prompt_embeds_2 = prompt_embeds_2[2313:]
        #    prompt_attention_masks = prompt_attention_masks[2313:]
        #    prompt_attention_masks_2 = prompt_attention_masks_2[2313:]
        elif not pixart and not coco_10k and not coco2014 and not hpsv2: # coco_1k
            prompt_embeds = prompt_embeds[:1000]
            prompt_embeds_2 = prompt_embeds_2[:1000]            
            prompt_attention_masks = prompt_attention_masks[:1000]
            prompt_attention_masks_2 = prompt_attention_masks_2[:1000]
            #def combine(array):
            #    part1 = array[0:700]
            #    part2 = array[1000:3300]
            #    return torch.cat((part1, part2), dim=0)
            #prompt_embeds = combine(prompt_embeds)
            #prompt_embeds_2 = combine(prompt_embeds_2)
            #prompt_attention_masks = combine(prompt_attention_masks)
            #prompt_attention_masks_2 = combine(prompt_attention_masks_2)
        negative_prompt_embeds = embedding_dict['negative_prompt_embeds'].to('cuda')
        negative_prompt_embeds_2 = embedding_dict['negative_prompt_embeds_2'].to('cuda')
        negative_prompt_attention_mask = embedding_dict['negative_prompt_attention_mask'].to('cuda')
        negative_prompt_attention_mask_2 = embedding_dict['negative_prompt_attention_mask_2'].to('cuda')
        return prompt_embeds, prompt_embeds_2, prompt_attention_masks, prompt_attention_masks_2, negative_prompt_embeds, negative_prompt_embeds_2, negative_prompt_attention_mask, negative_prompt_attention_mask_2
    else:
        print("File not found. Precomputing prompt embeddings...")
        if hpsv2:
            with open("/home/kgmills/coco_bs/hpsv2_prompts.txt", 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        elif pixart:
            with open(PIXART_TXT_FILE_LOC, 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        elif coco2014:
            with open(COCO_2014_CAPTIONS, 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        elif hpsv2:
            with open(HPSV2_CAPTIONS, 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        else:
            import json
            with open(COCO_VAL_CAPTIONS) as f:
                captions = json.load(f)
            captions_list = captions['annotations'][:10000]
            """
            if coco_10k:
                captions_list = captions_list[:10000]
            elif coco_9k:
                captions_list = captions_list[1000:10000]
            else:
                captions_list = captions_list[:1000]
            """
            prompt_list = [c['caption'] for c in captions_list]
            
        pes, pes2, pams, pams2, npe, npe2, npam, npam2 = [], [], [], [], None, None, None, None
        with torch.no_grad():
            for i, prompt in enumerate(prompt_list):
                print(i, prompt)
                if i == 0:
                    pe, npe, pam, npam = model.encode_prompt(prompt=prompt, text_encoder_index=0)
                    pe2, npe2, pam2, npam2 = model.encode_prompt(prompt=prompt, text_encoder_index=1)
                    
                else:
                    pe, _, pam, _ = model.encode_prompt(prompt=prompt, text_encoder_index=0)
                    pe2, _, pam2, _ = model.encode_prompt(prompt=prompt, text_encoder_index=1)
                    
                pes.append(pe.to('cpu'))
                pes2.append(pe2.to('cpu'))
                pams.append(pam.to('cpu'))
                pams2.append(pam2.to('cpu'))
            pes = torch.cat(pes, dim=0)
            pes2 = torch.cat(pes2, dim=0)
            pams = torch.cat(pams, dim=0)
            pams2 = torch.cat(pams2, dim=0)
            
        embedding_dict = {
            'prompt_embeds': pes,
            'prompt_embeds_2': pes2,
            'prompt_attention_masks': pams,
            'prompt_attention_masks_2': pams2,
            'negative_prompt_embeds': npe,
            'negative_prompt_embeds_2': npe2,
            'negative_prompt_attention_mask': npam,
            'negative_prompt_attention_mask_2': npam2
        }
        print("Saving precomputed prompt embeddings to disk: ", embedding_fname)
        torch.save(embedding_dict, embedding_fname)
        print("Save successful!")
        npe = npe.to("cuda")
        npam = npam.to("cuda")
        npe2 = npe2.to("cuda")
        npam2 = npam2.to("cuda")
        if coco_9k:
            # The number is change to restart the experiment
            return pes[1000:], pes2[1000:], pams[1000:], pams2[1000:], npe, npe2, npam, npam2
        elif coco2014:
            return pes[:5000], pes2[:5000], pams[:5000], pams2[:5000], npe, npe2, npam, npam2
        elif not pixart and not coco_10k and not coco2014 and not hpsv2:
            return pes[:1000], pes2[:1000], pams[:1000], pams2[:1000], npe, npe2, npam, npam2
        else:
            return pes, pes2, pams, pams2, npe, npe2, npam, npam2

            
if __name__ == "__main__":
    print("Testing")
    caption_list = get_captions(pixart=True)
    print(caption_list)
    print(len(caption_list))