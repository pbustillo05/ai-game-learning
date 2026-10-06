"""Bounded CUDA comparison of recorded-action ratio and full-action KL."""
from __future__ import annotations
import argparse, copy, json, math, time
from pathlib import Path
import torch
import yaml
from pbg.config import load_config_mapping
from pbg.core.learner import ReferenceState
from pbg.core.optimization import LossScaler
from pbg.runtime import TrainingSession
from experiments.training_diagnosis.probe_policies import OUT


def instrument(session, output):
    factory=session._iteration_engine
    def engine_factory():
        engine=factory(); prepare_base=engine._prepare_learner
        def prepare(owners,state,lane):
            prepared=prepare_base(owners,state,lane)
            current=session.modules.lane_module(lane)
            frozen=copy.deepcopy(current).eval().requires_grad_(False)
            learner=prepared.iteration.learner
            operator=learner.policy_operation.policy
            machine=learner.policy_operation.machine
            run_forward=learner.forward_loss_backward
            physical_base=prepared.iteration._physical_batches
            logical=[-1]
            def physical_batches(batch):
                logical[0]=int(batch.logical_step)
                yield from physical_base(batch)
            prepared.iteration._physical_batches=physical_batches
            def audited(batch,**kwargs):
                with torch.no_grad():
                    model_input=batch.sigma_fields['model_input']; action=batch.action_fields
                    with torch.autocast('cuda',dtype=torch.float16):
                        now=current.forward_partition(batch.input_type,model_input)
                        old=frozen.forward_partition(batch.input_type,model_input)
                    state=machine.state_from_model_input(batch.input_type,model_input,torch.zeros((batch.rows,machine.geometry.players),dtype=torch.int32,device=model_input.device))
                    legality=machine.legality(state)
                    p=oldp=operator.probabilities(old.policy_raw,legality)
                    q=operator.probabilities(now.policy_raw,legality)
                    eps=operator.epsilon
                    logp=p.clamp_min(eps).log();logq=q.clamp_min(eps).log()
                    # FP64 reduction preserves small full-KL values; model and policy stay production FP16/FP32.
                    full_forward=(p.double()*(logp.double()-logq.double())).sum(-1)
                    full_reverse=(q.double()*(logq.double()-logp.double())).sum(-1)
                    tv=(p-q).abs().sum(-1)*.5
                    mask=legality.event_ids.eq(action['action'][:,None])&legality.mask
                    col=mask.long().argmax(-1)
                    gather=col[:,None,None].expand(-1,p.shape[1],1)
                    op=p.gather(-1,gather).squeeze(-1);np=q.gather(-1,gather).squeeze(-1)
                    current_log=operator.selected_log_pi(now.policy_raw,legality,action['action'].to(torch.int32))
                    recomputed_old_log=operator.selected_log_pi(old.policy_raw,legality,action['action'].to(torch.int32))
                    stored_old_log=action['old_log_policy'].float()
                    logratio=current_log-stored_old_log;ratio=logratio.exp()
                    posterior=action['posterior'].float();posterior=posterior/posterior.sum(-1,keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
                    adv=action['advantage_target'].float()
                    diagnostic=posterior*((ratio-1)-logratio)
                    floor=stored_old_log.le(math.log(eps)+1e-5)
                    huge=ratio.gt(1e6)
                    positive=adv.gt(0);negative=adv.lt(0)
                    def mean_weighted(value):return float((posterior.double()*value.double()).sum(-1).mean())
                    metrics={
                        'recorded_action_approximate_kl':float(diagnostic.sum(-1).mean()),
                        'posterior_full_kl_old_current_floored':mean_weighted(full_forward),
                        'posterior_full_kl_current_old_floored':mean_weighted(full_reverse),
                        'posterior_total_variation':mean_weighted(tv),
                        'posterior_mass_stored_old_at_floor':mean_weighted(floor),
                        'posterior_mass_recomputed_old_exact_zero':mean_weighted(op.eq(0)),
                        'posterior_mass_recomputed_old_below_floor':mean_weighted(op.lt(eps)),
                        'posterior_mass_ratio_gt1e6':mean_weighted(huge),
                        'posterior_mass_ratio_gt1e6_positive_advantage':mean_weighted(huge&positive),
                        'posterior_mass_ratio_gt1e6_negative_advantage':mean_weighted(huge&negative),
                        'approximate_kl_from_ratio_gt1e6':float(diagnostic.masked_fill(~huge,0).sum(-1).mean()),
                        'approximate_kl_from_ratio_gt1e6_positive_advantage':float(diagnostic.masked_fill(~(huge&positive),0).sum(-1).mean()),
                        'approximate_kl_from_ratio_gt1e6_negative_advantage':float(diagnostic.masked_fill(~(huge&negative),0).sum(-1).mean()),
                        'posterior_old_log_abs_error':mean_weighted((recomputed_old_log-stored_old_log).abs()),
                    }
                    maxima={
                        'max_full_kl_old_current_floored':float(full_forward.max()),
                        'max_posterior_supported_full_kl_old_current_floored':float(full_forward.masked_fill(posterior<=0,0).max()),
                        'max_total_variation':float(tv.max()),
                        'max_ratio':float(ratio.max()),
                        'max_stored_recomputed_old_log_error':float((recomputed_old_log-stored_old_log).abs().max()),
                    }
                    top=[]
                    indices=torch.topk(diagnostic.flatten(),min(3,diagnostic.numel())).indices
                    for flat in indices.tolist():
                        row,hand=divmod(flat,diagnostic.shape[1])
                        if float(diagnostic[row,hand])<1:continue
                        top.append(dict(physical_row=row,retained_row=int(batch.indices[row]),hand=hand,action=int(action['action'][row]),posterior=float(posterior[row,hand]),advantage=float(adv[row,hand]),old_probability=float(op[row,hand]),current_probability=float(np[row,hand]),stored_old_log=float(stored_old_log[row,hand]),recomputed_old_log=float(recomputed_old_log[row,hand]),current_log=float(current_log[row,hand]),ratio=float(ratio[row,hand]),weighted_diagnostic=float(diagnostic[row,hand]),full_forward_kl=float(full_forward[row,hand]),full_reverse_kl=float(full_reverse[row,hand]),total_variation=float(tv[row,hand]),street=int(state.street[row]),actor=int(state.to_act[row]),stacks=state.stacks[row].tolist(),committed=state.committed[row].tolist(),board=state.board_cards[row].tolist(),current_type_logits=now.policy_raw['type_logits'][row,hand].tolist(),old_type_logits=old.policy_raw['type_logits'][row,hand].tolist()))
                    record=dict(logical_step=logical[0],family=batch.input_type,rows=batch.rows,objective_weight=float(kwargs.get('objective_weight',1)),means=metrics,maxima=maxima,top_counterexamples=top)
                    output['physical_batches'].append(record)
                    del now,old,p,oldp,q,logp,logq,full_forward,full_reverse,tv,op,np,current_log,recomputed_old_log,stored_old_log,logratio,ratio,posterior,adv,diagnostic
                return run_forward(batch,**kwargs)
            learner.forward_loss_backward=audited
            return prepared
        engine._prepare_learner=prepare
        return engine
    session._iteration_engine=engine_factory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=int,default=1100);parser.add_argument('--roots',type=int,default=512);parser.add_argument('--steps',type=int,default=4);parser.add_argument('--seed',type=int,default=918273)
    parser.add_argument('--aligned-completion', action='store_true')
    args=parser.parse_args();stem='nlhe_hu_32bb_rb_dualclip5_60h_20260814'
    path=Path('runs/experiments')/stem/f'checkpoint_{args.checkpoint:08d}.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    source=yaml.safe_load((Path('configs/experiments')/f'{stem}.yaml').read_text())
    source['scientific']['schedule']['steps']=args.steps
    run=source['run'];run['seed']=args.seed;run['workload']['roots']=args.roots;run['recovery']['minimum_roots']=args.roots;run['execution']['reference_cache']='off';run['iterations']=args.checkpoint+1
    run['output_dir']=str(OUT/f'ratio_audit_i{args.checkpoint}_seed{args.seed}')
    cfg=load_config_mapping(source);session=TrainingSession(cfg,initial_model_state=payload['model'],benchmark_telemetry=True)
    opt=session.optimizers.optimizers[0];opt.load_state_dict(payload['optimizer']);opt.scaler=LossScaler.from_state_dict(payload['scaler'])
    session.reference_modules.lane_module(0).load_state_dict(payload['reference_model'],strict=True)
    session.reference_states=(ReferenceState.from_state_dict(payload['reference_state']),);session.run_states[0].iteration=args.checkpoint
    result=dict(arguments=vars(args),checkpoint=str(path),physical_batches=[],notes='Full-action KL uses the same ActionTarget posterior across private hands, not a separately reconstructed pre-action state prior. Logs are floored at production epsilon1e-12; conditional zeros are separately counted.')
    instrument(session,result)
    if args.aligned_completion:
        from experiments.training_diagnosis.completion_alignment import install_completion_alignment
        result['completion_alignment']=[]
        install_completion_alignment(session,result['completion_alignment'])
    start=time.monotonic()
    try:
        session.step();result['seconds']=time.monotonic()-start;result['production_metrics']=copy.deepcopy(session.run_states[0].metrics_prefix[-1])
        steps={}
        for r in result['physical_batches']:
            d=steps.setdefault(r['logical_step'],dict(means={},maxima={},weight_sum=0,rows=0));d['weight_sum']+=r['objective_weight'];d['rows']+=r['rows']
            for k,v in r['means'].items():d['means'][k]=d['means'].get(k,0)+r['objective_weight']*v
            for k,v in r['maxima'].items():d['maxima'][k]=max(d['maxima'].get(k,-float('inf')),v)
        result['logical_steps']=steps
        result['iteration_means']={k:sum(v['means'][k] for v in steps.values())/len(steps) for k in next(iter(steps.values()))['means']}
        suffix='_aligned' if args.aligned_completion else ''
        destination=OUT/f'ratio_audit_rb_i{args.checkpoint}_seed{args.seed}{suffix}.json';destination.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'destination':str(destination),'seconds':result['seconds'],'iteration_means':result['iteration_means'],'logical_steps':steps},indent=2),flush=True)
    finally:session.close()

if __name__=='__main__':main()
