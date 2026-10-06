"""CUDA root-policy probes of the saved historical and current KL anchors."""
import gc
import json
from pathlib import Path

import torch
from pbg.inference import PokerInferenceSession
from experiments.training_diagnosis.probe_policies import install_legacy_preflop_inputs, summarize, OUT


def main():
    torch.set_num_threads(2)
    rb=Path('runs/experiments/nlhe_hu_32bb_rb_dualclip5_60h_20260814')
    template=torch.load(rb/'checkpoint_00000050.pt',map_location='cpu',weights_only=False,mmap=True)['scientific_config']
    points=[('old_fixed_reference','runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0001110.pt'),('new_initial_reference',str(rb/'checkpoint_00000050.pt')),('rb_final_reference',str(rb/'checkpoint_00010000.pt')),('normal_final_reference','runs/experiments/nlhe_hu_32bb_normal_60h_20260812/checkpoint_00010000.pt')]
    results=[]
    for name,path in points:
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        reference_iteration=payload.get('reference_iteration',payload.get('reference_state',{}).get('reference_iteration',None))
        session=PokerInferenceSession({'scientific_config':payload.get('scientific_config',template),'model':payload['reference_model'],'iteration':reference_iteration or 0,'source_checkpoint_sha256':'diagnostic-reference'},device='cuda:0')
        if 'scientific_config' not in payload:install_legacy_preflop_inputs(session)
        root=summarize(session,session.new_hand())
        row=dict(name=name,checkpoint=path,reference_iteration=reference_iteration,root=root)
        results.append(row)
        print(name,reference_iteration,'fold/call/raise',root['fold'],root['call_check'],root['raise'],'logits',root['raw_type_logits_min_by_type'],root['raw_type_logits_max_by_type'],flush=True)
        del session,payload
        gc.collect();torch.cuda.empty_cache()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'reference_audit.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
