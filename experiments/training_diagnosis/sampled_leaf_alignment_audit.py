"""CUDA integration proof: sampled terminal decision targets retain leaf identity."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from pbg.config import load_config_mapping
from pbg.core.learner import ReferenceState
from pbg.core.optimization import LossScaler
from pbg.runtime import TrainingSession
from pbg.games.poker.queued.hooks import SampledShowdownBinding
from experiments.training_diagnosis.probe_policies import OUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--aligned',action='store_true');args=parser.parse_args()
    stem='nlhe_hu_32bb_normal_60h_20260812';checkpoint=50
    payload=torch.load(Path('runs/experiments')/stem/f'checkpoint_{checkpoint:08d}.pt',map_location='cpu',weights_only=False,mmap=True)
    source=yaml.safe_load((Path('configs/experiments')/f'{stem}.yaml').read_text())
    source['scientific']['schedule']['steps']=1
    source['run']['seed']=472918
    source['run']['workload']['roots']=128
    source['run']['recovery']['minimum_roots']=128
    source['run']['execution']['reference_cache']='off'
    source['run']['iterations']=checkpoint+1
    source['run']['output_dir']=str(OUT/'sampled_identity_probe')
    session=TrainingSession(load_config_mapping(source),initial_model_state=payload['model'])
    optimizer=session.optimizers.optimizers[0];optimizer.load_state_dict(payload['optimizer']);optimizer.scaler=LossScaler.from_state_dict(payload['scaler'])
    session.reference_modules.lane_module(0).load_state_dict(payload['reference_model'],strict=True)
    session.reference_states=(ReferenceState.from_state_dict(payload['reference_state']),)
    session.run_states[0].iteration=checkpoint
    original_complete=SampledShowdownBinding.leaf_to_layer_work
    saved={};result={'checkpoint':checkpoint,'seed':472918,'roots':128,'aligned':args.aligned}
    def complete(binding,batch):
        plan=original_complete(binding,batch)
        for lane_index in range(batch.row.shape[0]):
            lane=int(batch.lanes[lane_index]);request=plan.destinations[0]
            for ordinal in range(batch.rows):
                leaf=int(batch.row[lane_index,ordinal])
                saved[leaf]={'actual_row':request.slab.tail(lane)+ordinal,'payoff':request.values['return_value'][lane,ordinal,0,:,0].clone()}
        return plan
    original_factory=session._iteration_engine
    def factory():
        engine=original_factory();old_targets=engine._execute_targets;old_prepare=engine._prepare_learner
        def targets(owners,state):
            plan=state.topology.plan
            result['movement']=[{'leaf':int(plan.movement_source_leaf_row[i]),'planned_row':int(plan.movement_destination_work_row[i]),'fifo_row':saved[int(plan.movement_source_leaf_row[i])]['actual_row']} for i in range(plan.movement_count)]
            result['movement_mismatch_count']=sum(r['planned_row']!=r['fifo_row'] for r in result['movement'])
            return old_targets(owners,state)
        def prepare(owners,state,lane):
            prepared=old_prepare(owners,state,lane);top=state.topology.topologies[lane];bindings=state.topology.bindings[lane];names=state.topology.plan.input_names
            byfamily={}
            for node in range(top.node_count):
                edge=int(top.child_begin[node])
                if int(top.target_kind[edge])!=1:continue
                leaf=int(top.target_row[edge]);owner=int(top.owner[node]);family=names[int(bindings.input_type[node])];row=int(top.sigma_head_row[node])
                byfamily.setdefault(family,[]).append((row,leaf,owner,node))
            checks=[]
            movement={r['leaf']:r for r in result['movement']}
            leafatrow={v['actual_row']:leaf for leaf,v in saved.items()}
            for family,rows in byfamily.items():
                gathered=prepared.sampler.stores[family].gather(torch.tensor([x[0] for x in rows],device='cuda'))
                actual=gathered.fields['value_target'].reshape(len(rows),-1)[:,0]
                for i,(row,leaf,owner,node) in enumerate(rows):
                    misplaced=leafatrow[movement[leaf]['planned_row']]
                    checks.append({'node':node,'family':family,'learner_row':row,'owner':owner,'correct_leaf':leaf,'leaf_at_unaligned_slot':misplaced,'actual_value_target':float(actual[i]),'correct_payoff':float(saved[leaf]['payoff'][owner]),'unrelated_leaf_payoff':float(saved[misplaced]['payoff'][owner])})
            result['terminal_decision_checks']=checks
            errors=[abs(r['actual_value_target']-r['correct_payoff']) for r in checks]
            unrelated=[abs(r['actual_value_target']-r['unrelated_leaf_payoff']) for r in checks]
            result.update(checked_terminal_decisions=len(checks),wrong_value_targets=sum(x>1e-7 for x in errors),max_value_target_error=max(errors),max_value_target_error_bb=max(errors)*32,mean_value_target_error=sum(errors)/len(errors),max_error_to_unrelated_leaf=max(unrelated))
            destination=OUT/f'sampled_leaf_alignment_{"aligned" if args.aligned else "baseline"}.json';destination.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('movement','terminal_decision_checks','alignment')},indent=2),flush=True)
            return prepared
        engine._execute_targets=targets;engine._prepare_learner=prepare;return engine
    session._iteration_engine=factory
    if args.aligned:
        from experiments.training_diagnosis.completion_alignment import install_completion_alignment
        result['alignment']=[];install_completion_alignment(session,result['alignment'])
    try:
        with patch.object(SampledShowdownBinding,'leaf_to_layer_work',complete):session.step()
    finally:session.close()


if __name__=='__main__':main()
