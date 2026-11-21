"""Diagnose B's fixed prompts at every token without changing RSS or the inputs."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from saebase.api import load_registered, encode_activations


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((ROOT/'assets/reproduction_inputs/stable_support/files/banana_ab_inputs.json').read_text())
    torch.set_num_threads(12);torch.manual_seed(0)
    pair=load_registered(model_key='llama31_8b',sae_key='llama31_8b_lxr_8x',layer=21,device='cpu',require_patch=False)
    raw=pair.model.model;device='cpu'
    free,_=torch.cuda.mem_get_info()
    if free>13*1024**3:
        device='cuda';raw.embed.to(device)
        for block in raw.blocks[:22]:block.to(device)
        raw.cfg.device=device;pair.sae.to(device)
    raw.eval()
    print('Diagnostic device:',device,flush=True)
    rows=[]
    for v,item in enumerate(inputs['groups']['B'],1):
        prompt=item['prompt'];ids=pair.model.tokens(prompt=prompt)
        with torch.inference_mode():
            h=pair.model._cached_sequence(ids,pair.site)
            z=encode_activations(pair.sae,h).float().cpu().numpy()
        encoded=raw.tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
        shift=len(ids[0])-len(encoded['input_ids']);end=len(item['prefix'])
        prefix_indices=[i+shift for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>0]
        boundary=max(prefix_indices)
        np.savez_compressed(out/f'B_V{v}_token_codes.npz',codes=z,tokens=ids.cpu().numpy())
        row={'variant':v,'topic':item['topic'],'prompt':prompt,'prefix':item['prefix'],'prefix_token_indices':prefix_indices,'prefix_last_token_index':boundary,'target_first_character':end+1,'target_last_token_index':len(ids[0])-1,'token_strings':[raw.tokenizer.decode([int(t)]) for t in ids[0].cpu().tolist()],'offset_mapping':encoded['offset_mapping'],'offset_token_shift':shift}
        rows.append(row)
        (out/'profiles.json').write_text(json.dumps({'device':device,'identity':pair.identity.to_dict(),'profiles':rows,'purpose':'Same B inputs, token-position diagnosis only. Prefix/content-token observations are not substituted into the original final-token RSS case.'},ensure_ascii=False,indent=2)+'\n')
        print('CAPTURED',v,item['topic'],'tokens',len(ids[0]),'prefix end',boundary,flush=True)


if __name__=='__main__':main()
