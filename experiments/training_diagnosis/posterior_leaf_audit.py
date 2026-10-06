"""Trace a suspicious RB action posterior to its completion leaf, CUDA only."""
import argparse,copy,json,math,time
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from pbg.config import load_config_mapping
from pbg.core.learner import ReferenceState
from pbg.core.optimization import LossScaler
from pbg.runtime import TrainingSession
from pbg.games.poker.queued.hooks import RBCompletionBinding
from experiments.training_diagnosis.probe_policies import OUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--aligned-completion',action='store_true');args=parser.parse_args()
    stem='nlhe_hu_32bb_rb_dualclip5_60h_20260814'; checkpoint=1100;row_wanted=1017;hand=126
    payload=torch.load(Path('runs/experiments')/stem/f'checkpoint_{checkpoint:08d}.pt',map_location='cpu',weights_only=False,mmap=True)
    source=yaml.safe_load((Path('configs/experiments')/f'{stem}.yaml').read_text());source['scientific']['schedule']['steps']=4
    run=source['run'];run['seed']=918273;run['workload']['roots']=512;run['recovery']['minimum_roots']=512;run['execution']['reference_cache']='off';run['iterations']=1101;run['output_dir']=str(OUT/'posterior_leaf_audit_i1100_seed918273')
    session=TrainingSession(load_config_mapping(source),initial_model_state=payload['model'],benchmark_telemetry=True)
    opt=session.optimizers.optimizers[0];opt.load_state_dict(payload['optimizer']);opt.scaler=LossScaler.from_state_dict(payload['scaler'])
    session.reference_modules.lane_module(0).load_state_dict(payload['reference_model'],strict=True);session.reference_states=(ReferenceState.from_state_dict(payload['reference_state']),);session.run_states[0].iteration=checkpoint
    saved=[];saved_work={};result={'source_checkpoint':checkpoint,'seed':918273,'roots':512,'target_row':row_wanted,'target_class':hand}
    original_complete=RBCompletionBinding.leaf_to_layer_work
    def completion(self,batch):
        saved.append((batch.row.clone(),{k:v.clone() for k,v in batch.fields.items()},self.evaluator))
        plan=original_complete(self,batch)
        for lane_index in range(batch.row.shape[0]):
            lane_id=int(batch.lanes[lane_index])
            for source_index in range(batch.rows):
                valid=plan.coverage.valid[lane_id]&plan.coverage.source_ordinal[lane_id].eq(source_index)
                descriptor=int(torch.nonzero(valid).flatten()[0]);destination=int(plan.coverage.destination[lane_id,descriptor]);ordinal=int(plan.coverage.destination_ordinal[lane_id,descriptor])
                request=plan.destinations[destination];leaf_id=int(batch.row[lane_index,source_index]);actual_row=request.slab.tail(lane_id)+ordinal
                saved_work[leaf_id]={'output':request.input_type,'actual_row':actual_row,'mass':request.values['mass'][lane_id,ordinal,0].clone()}
        return plan
    engine_base=session._iteration_engine
    def factory():
        engine=engine_base();prepare_base=engine._prepare_learner
        targets_base=engine._execute_targets
        def targets(owners,state):
            plan=state.topology.plan;comparison=[]
            for i in range(plan.movement_count):
                leaf=int(plan.movement_source_leaf_row[i]);planned=int(plan.movement_destination_work_row[i]);output=plan.output_names[int(plan.movement_output_type[i])];actual=saved_work[leaf]
                comparison.append({'leaf':leaf,'output':output,'planned_row':planned,'actual_row':actual['actual_row']})
            result['movement_mismatch_count']=sum(x['planned_row']!=x['actual_row'] for x in comparison);result['movement_total']=len(comparison)
            result['movement_rows']=comparison
            return targets_base(owners,state)
        engine._execute_targets=targets
        def prepare(owners,state,lane):
            prepared=prepare_base(owners,state,lane);top=state.topology.topologies[0];binding=state.topology.bindings[0]
            names=state.topology.plan.input_names;pre=names.index('pre')
            matches=torch.nonzero((binding.input_type==pre)&(top.sigma_head_row==row_wanted)).flatten().tolist()
            if len(matches)!=1:raise RuntimeError(f'nonunique node {matches}')
            node=matches[0];ancestry=[];n=node
            while n>=0:ancestry.append(n);n=int(top.parent[n])
            ancestry.reverse();desc=[];n=node
            while True:
                desc.append(n);edge=int(top.child_begin[n]);kind=int(top.target_kind[edge]);target=int(top.target_row[edge])
                if kind==1:leaf=target;break
                n=target
            path=ancestry+desc[1:];result['topology_node']=node;result['leaf_row']=leaf;result['path']=[]
            selected_movement=next(x for x in result['movement_rows'] if x['leaf']==leaf)
            wrong_leaf=next(key for key,val in saved_work.items() if val['output']==selected_movement['output'] and val['actual_row']==selected_movement['planned_row'])
            result['selected_leaf_movement']={**selected_movement,'leaf_actually_at_planned_row':wrong_leaf}
            model=session.modules.lane_module(0);policy=prepared.iteration.learner.policy_operation.policy;machine=prepared.iteration.learner.policy_operation.machine
            classes=model.combo_class;cards=machine.combo_cards(torch.arange(machine.geometry.private_hands,device='cuda',dtype=torch.int32))
            selected_fields=None;evaluator=None
            for rowids,fields,ev in saved:
                hits=torch.nonzero(rowids==leaf)
                if hits.numel():
                    a,b=hits[0].tolist();selected_fields={k:v[a,b:b+1] for k,v in fields.items()};evaluator=ev;break
            if selected_fields is None:raise RuntimeError('leaf not found in completion packs')
            with torch.no_grad():
                for n in path:
                    famidx=int(binding.input_type[n]);fam=names[famidx] if famidx>=0 else None;rr=int(top.sigma_head_row[n]);entry={'node':n,'family':fam,'row':rr,'owner':int(top.owner[n]),'edge_label':int(top.label[int(top.child_begin[n])])}
                    if fam is not None:
                        batch=prepared.sampler.stores[fam].gather(torch.tensor([rr],device='cuda'));fields=batch.fields
                        st=machine.state_from_model_input(fam,fields['model_input'],torch.zeros((1,2),dtype=torch.int32,device='cuda'));reach,history=machine.exact_model_context(fam,fields['model_input'])
                        with torch.autocast('cuda',dtype=torch.float16):evaluation=model.forward_partition(fam,fields['model_input'])
                        probs=policy.probabilities(evaluation.policy_raw,machine.legality(st));col=(machine.legality(st).event_ids==fields['action'][:,None]).long().argmax(-1);selected=probs.gather(-1,col[:,None,None].expand(-1,probs.shape[1],1)).squeeze(-1)
                        posterior=fields['posterior'];posterior=posterior/posterior.sum(-1,keepdim=True)
                        pr=[];po=[]
                        for seat in range(2):
                            weights=reach[0,seat].softmax(-1);pr.append(float(weights[classes==hand].sum()))
                        entry.update(action=int(fields['action'][0]),street=int(st.street[0]),actor=int(st.to_act[0]),stacks=st.stacks[0].tolist(),committed=st.committed[0].tolist(),board=st.board_cards[0].tolist(),input_class126_prior_by_seat=pr,history_actor=history.actor[0][history.mask[0]].tolist(),history_action=history.action[0][history.mask[0]].tolist())
                        if fam.startswith('post'):entry['class126_action_posterior']=float(posterior[0,classes==hand].sum())
                        if fam=='pre':entry.update(class126_action_probability=float(selected[0,hand]),class126_old_log=float(fields['old_log_policy'][0,hand]),class126_action_posterior=float(posterior[0,hand]),class126_advantage=float(fields['advantage_target'][0,hand]),class126_type_logits=evaluation.policy_raw['type_logits'][0,hand].tolist())
                        if n==node:action_posterior=posterior[0].clone();actor=int(st.to_act[0]);target_input_reach=reach[0].clone()
                    result['path'].append(entry)
                f=selected_fields;w=f['range_weights'][0].float();board=f['leaf_public_state'][0]
                boardlegal=~((cards[:,:,None]==board[None,None,:])&(board[None,None,:]!=255)).any(-1).any(-1)
                compatibility=~(cards[:,None,:,None]==cards[None,:,None,:]).any(-1).any(-1)
                w=w*boardlegal[None]
                raw0=w[0]*(compatibility.float()@w[1]);raw1=w[1]*(compatibility.float().T@w[0]);exact=torch.stack((raw0/raw0.sum(),raw1/raw1.sum()))
                kwargs=dict(board_cards=f['leaf_public_state'].contiguous(),range_weights=f['range_weights'].contiguous(),stratum_level=f['stratum_level'].contiguous(),payoff_baseline=f['payoff_baseline'].contiguous(),stratum_amount=f['stratum_amount'].contiguous(),n_samples=1024,seed=f['seed'].reshape(1).contiguous())
                evaluated=evaluator.complete(**kwargs);empirical=evaluated.p_hat[0]
                exact_class=torch.zeros((2,169),device='cuda').scatter_add_(1,classes[None].expand(2,-1),exact)
                empirical_class=torch.zeros((2,169),device='cuda').scatter_add_(1,classes[None].expand(2,-1),empirical)
                original_mass=saved_work[leaf]['mass'];wrong_mass=saved_work[wrong_leaf]['mass']
                if original_mass.shape[-1]==1326:original_class=torch.zeros((2,169),device='cuda').scatter_add_(1,classes[None].expand(2,-1),original_mass)
                else:original_class=original_mass
                if wrong_mass.shape[-1]==1326:wrong_class=torch.zeros((2,169),device='cuda').scatter_add_(1,classes[None].expand(2,-1),wrong_mass)
                else:wrong_class=wrong_mass
                result['wrong_leaf_mass_comparison']={'wrong_leaf':wrong_leaf,'action_vs_wrong_leaf_max_abs':float((action_posterior-wrong_class[actor]).abs().max()),'wrong_leaf_class126_mass_by_seat':wrong_class[:,hand].tolist(),'action_vs_original_leaf_max_abs':float((action_posterior-original_class[actor]).abs().max()),'original_vs_replayed_evaluator_max_abs':float((original_class-empirical_class).abs().max())}
                result['leaf']={'board':board.tolist(),'actor':actor,'stratum_level':f['stratum_level'][0].tolist(),'payoff_baseline':f['payoff_baseline'][0].tolist(),'stratum_amount':f['stratum_amount'][0].tolist(),'seed':int(f['seed'].reshape(-1)[0]),'range_class126_unnormalized_by_seat':[float(w[s,classes==hand].sum()) for s in range(2)],'range_mass_by_seat':w.sum(-1).tolist(),'range_class126_normalized_by_seat':[float((w[s]/w[s].sum())[classes==hand].sum()) for s in range(2)],'exact_class126_posterior_by_seat':exact_class[:,hand].tolist(),'evaluator_class126_posterior_by_seat':empirical_class[:,hand].tolist(),'action_class126_posterior':float(action_posterior[hand]),'action_vs_evaluator_max_abs':float((action_posterior-empirical_class[actor]).abs().max()),'action_vs_exact_l1':float((action_posterior-exact_class[actor]).abs().sum()),'evaluator_vs_exact_l1_by_seat':(empirical_class-exact_class).abs().sum(-1).tolist(),'evaluator_n_eff':float(evaluated.n_eff[0]),'exact_actor_top_classes':[{'class':int(h),'probability':float(exact_class[actor,h])} for h in torch.topk(exact_class[actor],10).indices.tolist()]}
            suffix='_aligned' if args.aligned_completion else ''
            result['aligned_completion']=args.aligned_completion
            destination=OUT/f'posterior_leaf_audit_rb_i1100_seed918273{suffix}.json';destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
            return prepared
        engine._prepare_learner=prepare;return engine
    session._iteration_engine=factory
    if args.aligned_completion:
        from experiments.training_diagnosis.completion_alignment import install_completion_alignment
        result['alignment']=[]
        install_completion_alignment(session,result['alignment'])
    try:
        with patch.object(RBCompletionBinding,'leaf_to_layer_work',completion):session.step()
    finally:session.close()

if __name__=='__main__':main()
