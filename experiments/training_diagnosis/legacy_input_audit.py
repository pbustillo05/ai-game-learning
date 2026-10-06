"""CUDA diagnosis of preflop legacy checkpoint/input compatibility.

Historical modules are verbatim 537700f copies in /tmp, with import prefixes
renamed only to permit loading both historical and present networks together.
The historical rollout computed max-gauged exact-hand reaches before class
summing; the queued model gauges class-summed reaches. Both forms are probed.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
import subprocess

import torch

legacy_root = Path('/tmp/pbg_diagnosis_legacy537')
for relative in ('cards.py', 'folded.py', 'model/preflop.py', 'model/postflop.py', 'model/heads.py', 'model/incidence.py', 'model/primitives.py'):
    destination = legacy_root / 'legacy_poker' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    historical_source = subprocess.check_output(['git', 'show', '537700f:pbg/games/poker/' + relative], text=True)
    destination.write_text(historical_source.replace('pbg.games.poker', 'legacy_poker'))
for relative in ('__init__.py', 'model/__init__.py'):
    (legacy_root / 'legacy_poker' / relative).write_text('')
sys.path.insert(0, str(legacy_root))
from legacy_poker.model.preflop import PreflopNet as OldPreflopNet
from legacy_poker.model.preflop import PokerModelConfig as OldConfig
from legacy_poker.model.preflop import PokerPublicBatch as OldPublic
from legacy_poker.model.preflop import PackedPublicHistory as OldHistory
from legacy_poker.model.preflop import PreflopMasks as OldMasks
from legacy_poker.model.primitives import MHA as OldMHA
from legacy_poker.model.postflop import PostflopNet as OldPostflopNet
from legacy_poker.model.postflop import PostflopMasks as OldPostflopMasks
from pbg.games.poker.model.postflop import PostflopMasks
from pbg.games.poker.model.preflop import PreflopMasks
from pbg.inference import PokerInferenceSession

OUT = Path('runs/experiments/nlhe_hu_32bb_rb_training_diagnosis_20260925')


def main():
    torch.set_num_threads(2)
    payload = torch.load('runs/nlhe_hu_32bb_full_sequential_rb/ckpt_0001110.pt', map_location='cpu', weights_only=False, mmap=True)
    template = torch.load('runs/experiments/nlhe_hu_32bb_rb_dualclip5_60h_20260814/checkpoint_00010000.pt',map_location='cpu',weights_only=False,mmap=True)['scientific_config']
    session = PokerInferenceSession({'scientific_config': template, 'model': payload['model'], 'iteration':1110,'source_checkpoint_sha256':'diagnostic'},device='cuda:0')
    old_net = OldPreflopNet(OldConfig(**payload['config']['model']), geometry=13).to('cuda:0').eval()
    old_net.load_state_dict({k[len('preflop.'):]:v for k,v in payload['model'].items() if k.startswith('preflop.')},strict=True)
    for m in old_net.modules():
        if isinstance(m,OldMHA):m.configure_execution('sdpa',128)
    hand = session.new_hand()
    results = []
    with torch.inference_mode():
        for spot in ('root','bb_facing_3bb'):
            item = session._hand(hand)
            pub,_ = session.model._public(item.state,item.history)
            old_fields = {f.name:getattr(pub,f.name) for f in dataclasses.fields(OldPublic)}
            old_fields['history'] = OldHistory(**{f.name:getattr(pub.history,f.name) for f in dataclasses.fields(OldHistory)})
            old_pub = OldPublic(**old_fields)
            finite = torch.isfinite(item.log_reach)
            weights = (item.log_reach - item.log_reach.masked_fill(~finite,-torch.inf).amax(-1,keepdim=True)).exp().masked_fill(~finite,0)
            ranges = weights.new_zeros(1,2,169)
            ranges.scatter_add_(2,session.model.combo_class[None,None,:].expand_as(weights),weights)
            legacy_output = old_net(old_pub,ranges,pub.to_act,OldMasks(seat_mask=pub.seat_mask,status=pub.status))
            current_same_input = session.model.preflop(pub,ranges,pub.to_act,PreflopMasks(seat_mask=pub.seat_mask,status=pub.status))
            current_eval,legal,current_p = session._evaluate(item)
            legacy_p = session.policy.probabilities(legacy_output.raw,legal)
            fixed_input_p = session.policy.probabilities(current_same_input.raw,legal)
            actor=int(item.state.to_act[0])
            class_mass=weights[0,actor].new_zeros(169)
            class_mass.scatter_add_(0,session.model.combo_class,weights[0,actor])
            class_mass/=class_mass.sum()
            def summary(prob):
                aggregate = class_mass @ prob[0]
                return {'fold':float(aggregate[0]),'call':float(aggregate[1]),'raise':float(aggregate[2:].sum()),'raise_std':float(prob[0,:,2:].sum(-1).std())}
            results.append({'spot':spot,'historical_input_class_max':ranges.amax(-1).tolist(),'historical_input_class_sum':ranges.sum(-1).tolist(),'historical_network_historical_input':summary(legacy_p),'current_network_historical_input':summary(fixed_input_p),'current_network_current_input':summary(current_p),'network_same_input_probability_max_error':float((legacy_p-fixed_input_p).abs().max()),'network_same_input_value_max_error':float((legacy_output.v-current_same_input.v).abs().max()),'current_input_probability_max_error':float((legacy_p-current_p).abs().max())})
            # To preserve historical public-reach update semantics, advance with
            # a temporary preflop wrapper that restores exact-gauged class scale.
            original = session.model._call_preflop
            def historical_call(model_input):
                state=session.model._state('pre',model_input)
                log_class,hist=session.machine.model_context('pre',model_input)
                pp,_=session.model._public(state,hist)
                exact_log=session.machine.expand_preflop_reach(log_class)
                exact_weight=(exact_log-exact_log.amax(-1,keepdim=True)).exp()
                rr=exact_weight.new_zeros(exact_weight.shape[:2]+(169,))
                rr.scatter_add_(2,session.model.combo_class[None,None,:].expand_as(exact_weight),exact_weight)
                return session.model._evaluation(session.model.preflop(pp,rr,pp.to_act,PreflopMasks(seat_mask=pp.seat_mask,status=pp.status)))
            session.model._call_preflop=historical_call
            if spot=='root':session.advance(hand,('raise_to',3.0))
            else:
                session.advance(hand,'call')
                session.deal(hand,['As','7h','2s'])
            session.model._call_preflop=original
    old_post = OldPostflopNet(OldConfig(**payload['config']['model']), geometry=13).to('cuda:0').eval()
    old_post.load_state_dict({k[len('postflop.'):]:v for k,v in payload['model'].items() if k.startswith('postflop.')},strict=True)
    for m in old_post.modules():
        if isinstance(m,OldMHA):m.configure_execution('sdpa',128)
    with torch.inference_mode():
        item=session._hand(hand)
        pub,compatible=session.model._public(item.state,item.history)
        fields={f.name:getattr(pub,f.name) for f in dataclasses.fields(OldPublic)}
        fields['history']=OldHistory(**{f.name:getattr(pub.history,f.name) for f in dataclasses.fields(OldHistory)})
        old_pub=OldPublic(**fields)
        finite=torch.isfinite(item.log_reach)
        ranges=(item.log_reach-item.log_reach.masked_fill(~finite,-torch.inf).amax(-1,keepdim=True)).exp().masked_fill(~finite,0)
        common=dict(compatibility=compatible,seat_mask=pub.seat_mask,combo_mask=ranges.bool(),status=pub.status,live_seat_ids=torch.arange(2,device='cuda:0')[None])
        old_output=old_post(old_pub,ranges,pub.to_act,OldPostflopMasks(**common))
        current_output=session.model.postflop(pub,ranges,pub.to_act,PostflopMasks(**common))
        current_eval,legal,current_p=session._evaluate(item)
        old_p=session.policy.probabilities(old_output.raw,legal)
        direct_p=session.policy.probabilities(current_output.raw,legal)
        weights=item.log_reach[0,int(item.state.to_act[0])].softmax(-1)
        def summary(prob):
            aggregate=weights@prob[0]
            return dict(fold=float(aggregate[0]),call=float(aggregate[1]),raise_probability=float(aggregate[2:].sum()))
        results.append(dict(spot='flop_A72_two_spades',historical_network=summary(old_p),current_network=summary(current_p),same_input_probability_max_error=float((old_p-direct_p).abs().max()),inference_probability_max_error=float((old_p-current_p).abs().max()),same_input_value_max_error=float((old_output.v-current_output.v).abs().max())))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'legacy_input_audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
