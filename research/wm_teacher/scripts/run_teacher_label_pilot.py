#!/usr/bin/env python3
"""One-shot, four-node label pilot, not a semantic or Teacher quality claim."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import prepare_teacher_label_admission as prep
import run_teacher_generation as process
import recaption_e3 as caption

ROOT=Path(__file__).resolve().parents[1]
LOCAL=Path('/dockerdata/wm_tmp/teacher_label_pilot_20261102_v3_recovery')
OLD_LOCAL_ROOTS=(Path('/dockerdata/wm_tmp/teacher_label_pilot_20261102'),
                 Path('/dockerdata/wm_tmp/teacher_label_pilot_20261102_attempt2'))
ATTEMPT_ID='label_v3_one_recovery'
PRIOR_COST_SECONDS={0:18.143487762659788,1:15.434166906867176,2:19.951568322256207,3:17.300606979057193}
PRIOR_FIRST_SHA256=('812647e2a361f5eb9627bb9bed307efd29f08d28601391b7ed595dce0a9a6192',
                    '4c5eecc41dc62bdd73f09f77183d7c4aada93b20e744f223c2c393f04b337b3d',
                    '895b9e95dc5449b32c55b4ce74f2c4586ea751f486d268fbb14e476d45c3ab85',
                    'e6075db49e35d158d53c90eded4e4c408385480d88835e7812f53b94a53c95f8')
EVIDENCE_ROOT=ROOT/'eval/teacher_label_pilot_20261102'
AUTH=ROOT/'eval/diag/teacher_training_admission_authorization_20261101.json'
AUTH_SHA256='245e14b3b7a4d3073c6914b7c9c8fba8907261aa85b6f6a3e9a59be7f400c2d9'
RECOVERY_AUTH=EVIDENCE_ROOT/'v3_one_recovery.authorization.json'
RECOVERY_AUTH_SHA256='ccc8b7cf3e72e37b4bc32634f6294eeadd0d11a1c26b921e0265e823928b868b'
OLD_PLAN=EVIDENCE_ROOT/'plan.lease_fix.json'
OLD_PLAN_SHA256='38236595a602b8e91946d76ddfe07d4cab79c569fd6c40c77714c390ab87d2ad'
COMPLETED_AUDIT=EVIDENCE_ROOT/'attempt2_collected/completion.audit.json'
COMPLETED_AUDIT_SHA256='46849ebb36d7cfcc9779532ee9b4e7f72e603a2df11de1d5274c8bde900fe860'
MANIFEST=ROOT/'training_metadata/e3_recaption_20261022/pool1856_dual_manifest.jsonl'
SCHEMA='teacher_label_pilot_execution_v3'
CAP=810
TAIL=90
MODEL_ROOT=Path('/dockerdata/models_qwen25_local_20261023')
MODEL_FILES=tuple(f'model-{i:05d}-of-00005.safetensors' for i in range(1,6))+('model.safetensors.index.json','config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','chat_template.json')
SOURCE_NAMES=('scripts/run_teacher_label_pilot.py','tests/test_teacher_label_pilot.py',
              'scripts/prepare_teacher_label_admission.py','scripts/recaption_e3.py',
              'scripts/run_teacher_generation.py','scripts/prepare_teacher_recovery.py',
              'scripts/run_teacher_inference_smoke.py','scripts/gpu_hold.py',
              'scripts/run_teacher_devset.py','scripts/measure_teacher_throughput.py',
              'scripts/prepare_teacher_comparison.py','scripts/run_teacher_base.py','scripts/verify_teacher_history.py')
require=prep.require


def encoded(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False)+chr(10)).encode()


def record(path,nonempty=True):
    path=Path(path).absolute();before=path.stat();digest=hashlib.sha256()
    require(path.resolve()==path and path.is_file() and (not nonempty or before.st_size>0),'Non-symlink file required')
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'File changed while hashing')
    return dict(path=str(path),bytes=before.st_size,sha256=digest.hexdigest())


def check(row):
    require(record(row['path'],nonempty=row['bytes']>0)==row,'Current content differs from approved bytes')


def read(row):
    raw=Path(row['path']).read_bytes()
    require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Approved JSON byte drift')
    return prep.strict_json(raw)


def write_new(path,value):
    path=Path(path);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
    return record(path)


def runtime():
    require('torch' not in sys.modules,'Prepare must not import CUDA/runtime models')
    packages={}
    rows={}
    for name,module in (('torch','torch'),('transformers','transformers'),('safetensors','safetensors'),('pillow','PIL'),('numpy','numpy'),('tokenizers','tokenizers'),('accelerate','accelerate')):
        d=importlib.metadata.distribution(name);packages[name]=d.version
        paths=[Path(d.locate_file(module+'/__init__.py')).resolve()]
        if name=='torch':paths+=list(Path(d.locate_file('torch')).glob('_C*.so'))
        for p in paths:rows[str(p)]=record(p,nonempty=p.stat().st_size>0)
    tools={n:shutil.which(n) for n in ('ffmpeg','timeout','nvidia-smi')}
    require(all(tools.values()),'Runtime tools missing')
    for p in list(tools.values())+[str(Path(sys.executable).resolve())]:rows[str(Path(p).resolve())]=record(Path(p).resolve())
    return dict(python=str(Path(sys.executable).resolve()),packages=packages,tools=tools,files=list(rows.values()),
        caveat='Selected package entrypoints/binary/tools, not exhaustive installed environment attestation')


def canonical_models(proof):
    return [{k:r[k] for k in ('relative','bytes','sha256')} for r in proof['models']]


def validate_model_proof(proof,rank):
    require(type(proof.get('rank')) is int and proof['rank']==rank and proof['schema']=='teacher_label_model_content_v1'
            and proof['GPU_used'] is False and proof['model_type']=='qwen2_5_vl' and proof['weight_shards']==5,
            'Actual typed Qwen model proof required')
    require(proof.get('model_root')==str(MODEL_ROOT) and MODEL_ROOT.resolve()==MODEL_ROOT,
            'Only the approved local Qwen loading root is allowed')
    rows=proof['models'];require([r['relative'] for r in rows]==list(MODEL_FILES),'Exact five weight shards and nine configs/tokenizers required')
    for row in rows:
        require(row['path']==str(MODEL_ROOT/row['relative']) and type(row['bytes']) is int and row['bytes']>0
                and len(row['sha256'])==64,'Model hash path must be the actual loaded local file')
    identity=canonical_models(proof)
    require(hashlib.sha256(encoded(identity)).hexdigest()==proof['content_sha256'],'Model content tuple digest mismatch')
    return proof


def pinned(path,digest):
    row=record(path);require(row['sha256']==digest,'Immutable original evidence SHA drift')
    return row


def validate_authorizations(auth,recovery):
    require(auth==pinned(AUTH,AUTH_SHA256) and recovery==pinned(RECOVERY_AUTH,RECOVERY_AUTH_SHA256),
            'Original and one-recovery authorization bindings required')
    original=read(auth);contract=read(recovery)
    require(original['schema']=='teacher_training_admission_authorization_v1'
            and original['stage_caps_GPUh']=={'label_rebuild_and_cache':24,'single_gpu_training_resume':2,'multi_node_technical_validation':6}
            and type(original['total_cap_GPUh']) is int and original['total_cap_GPUh']==32
            and all(original[k] is False for k in ('formal_candidate_training_budget_approved','phase_borrowing_allowed',
                'AR_allowed','student_or_distillation_allowed','previous_experiment_budgets_reused','training_started'))
            and original['failed_allocations_charged'] is True,'Training admission authorization drift')
    expected=dict(schema='teacher_label_v3_one_recovery_authorization_v1',user_choice='批准一次v3恢复',
        original_training_authorization_path=str(AUTH),original_training_authorization_sha256=AUTH_SHA256,
        prompt_version='e3-grounded-recaption-v3',fixed_sample_count=16,same_selection_seed=20261101,
        same_previously_selected_cases=True,allocated_gpus_per_node=1,gpu_index=0,nodes=4,cap_seconds_per_node=810,
        pilot_total_cap_seconds_including_all_prior=3600,previous_failed_seconds=70.82982997084036,
        previous_completed_seconds=259.7616256037727,all_previous_costs_preserved=True,
        charged_to_stage='label_rebuild_and_cache',stage_cap_GPUh=24,new_directory_required=True,
        new_source_and_publication_gate_required=True,independent_review_required=True,recovery_attempts_allowed=1,
        automatic_retry_allowed=False,full_label_rebuild_allowed=False,formal_training_allowed=False,
        AR_allowed=False,student_distillation_allowed=False)
    require(set(contract)==set(expected)|{'authorized_at'} and isinstance(contract['authorized_at'],str)
            and all(type(contract[k]) is type(v) and contract[k]==v for k,v in expected.items()),
            'Single v3 recovery authorization schema/scope drift')
    require(caption.PROMPT_VERSION==expected['prompt_version'] and prep.SEED==expected['same_selection_seed']
            and CAP==810 and TAIL==90 and LOCAL not in OLD_LOCAL_ROOTS and LOCAL.resolve()==LOCAL,
            'Only the fresh v3 recovery root/prompt/cap is permitted')
    return contract


def freeze_history():
    old_plan=pinned(OLD_PLAN,OLD_PLAN_SHA256);old=read(old_plan)
    require(old['schema']=='teacher_label_pilot_execution_v2' and old['attempt_id']=='label_pilot_attempt2'
            and old['prompt_version']=='e3-grounded-recaption-v2'
            and old['selection_seed']==20261101 and len(old['jobs'])==16,
            'Original attempt2 identity/selection required, not current-source verification')
    audit_row=pinned(COMPLETED_AUDIT,COMPLETED_AUDIT_SHA256);audit=read(audit_row)
    require(audit['schema']=='teacher_label_pilot_actual_completion_v1' and audit['plan']==old_plan
            and audit['labels_written']==16 and audit['error_records']==0 and audit['label_stage_cap_GPUh']==24
            and audit['pilot_cap_GPUh']==1 and audit['semantic_review_pass'] is False
            and audit['formal_training_started'] is False and len(audit['nodes'])==4
            and [r['index'] for r in audit['records']]==list(range(16)), 'Original completed pilot audit drift')
    failed=[];completed=[];first_cost=[];second_cost=[]
    for rank in range(4):
        first_row=pinned(EVIDENCE_ROOT/f'attempt1_rank{rank}.json',PRIOR_FIRST_SHA256[rank]);first=read(first_row)
        require(type(first['cost']['rank']) is int and first['cost']['rank']==rank and first['parent']['rank']==rank
                and first['cost']['charged_upper_seconds']==PRIOR_COST_SECONDS[rank]
                and first['cost']['release_verified'] is True and first['cost']['budget_exceeded'] is False
                and first['parent']['released'] is True and first['parent']['success'] is False
                and first['CUDA_initialized'] is False and first['labels_written']==0,'Original failed allocation charges/release required')
        failed.append(first_row);first_cost.append(first['cost']['charged_upper_seconds'])
        entry=audit['nodes'][rank];require(type(entry['rank']) is int and entry['rank']==rank and entry['released'] is True,
                                        'Original completed node ordering/release drift')
        base=EVIDENCE_ROOT/f'attempt2_collected/rank{rank}'
        cost_row=record(base/'cost.json');parent_row=record(base/'parent.json');worker_row=record(base/'worker.json')
        require(entry['cost']==cost_row and entry['parent']==parent_row,'Original audit cost/parent raw-byte binding drift')
        cost=read(cost_row);parent=read(parent_row);worker_value=read(worker_row)
        require(cost['plan']==old_plan and type(cost['rank']) is int and cost['rank']==rank
                and cost['release_verified'] is True and cost['budget_exceeded'] is False
                and type(cost['charged_upper_seconds']) in (int,float) and 0<cost['charged_upper_seconds']<=860
                and cost['charged_upper_seconds']==cost['elapsed_seconds']==entry['charged_seconds']==entry['actual_seconds'],
                'Original raw completed cost/audit mismatch')
        require(parent['schema']=='teacher_label_pilot_parent_v1' and parent['plan']==old_plan and parent['rank']==rank
                and parent['released'] is True and parent['success'] is True and type(parent['exit_code']) is int and parent['exit_code']==0
                and parent['error'] is None and parent['interrupted_signals']==[] and parent['cap_seconds']==860
                and parent['semantic_review_pass'] is False and parent['formal_training_allowed'] is False
                and len(parent['release_queries'])==2 and all(q['pids']==[] and q['returncode']==0 for q in parent['release_queries']),
                'Original parent successful exit/release required')
        require(parent['worker']==dict(worker_row,path=str(OLD_LOCAL_ROOTS[1]/'worker.json'))
                and worker_value['plan']==old_plan and worker_value['rank']==rank and worker_value['success'] is True
                and worker_value['semantic_review_pass'] is False and worker_value['formal_training_allowed'] is False
                and len(worker_value['records'])==4, 'Original worker bytes/exit/full coverage required')
        labels=[]
        for offset,index in enumerate(range(rank*4,rank*4+4)):
            entry_label=audit['records'][index];label_row=record(base/f'label{index}.json');label=read(label_row);job=old['jobs'][index]
            require(entry_label['label']==label_row and entry_label['video']==job['video']
                    and entry_label['semantic_review_pass'] is False and label==worker_value['records'][offset]
                    and 'error_type' not in label and label['sample_id']==job['sample_id']
                    and label['video_sha256']==job['video']['sha256'] and label['original_prompt_sha256']==job['original_prompt_sha256']
                    and label['prompt_version']=='e3-grounded-recaption-v2' and label['semantic_review_pass'] is False,
                    'Original label bytes/ordered identity coverage drift')
            # Historical v2 facts are preserved, never revalidated as v3 or promoted to a semantic pass.
            labels.append(label_row)
        completed.append(dict(rank=rank,cost=cost_row,parent=parent_row,worker=worker_row,labels=labels))
        second_cost.append(cost['charged_upper_seconds'])
    first_total=sum(first_cost);second_total=sum(second_cost)
    require(first_total==70.82982997084036==audit['previous_failed_seconds']
            and second_total==259.7616256037727==audit['new_allocated_seconds']
            and (first_total+second_total)/3600==audit['cumulative_allocated_GPUh']
            and first_total+second_total+4*CAP<=3600,'Both original pilot costs must remain inside the original reservation')
    return dict(original_attempt2_plan=old_plan,prior_failed_attempt_proofs=failed,
        prior_completed_attempt_audit=audit_row,prior_completed_attempt_rank_proofs=completed,
        prior_failed_charged_seconds=first_total,prior_completed_charged_seconds=second_total,
        prior_attempt_charged_seconds=math.fsum(first_cost+second_cost))


def validate_same_jobs(jobs,old):
    keys=('index','rank','sample_id','video','original_prompt_sha256')
    require(len(jobs)==16 and len(old['jobs'])==16
            and all({k:j[k] for k in keys}=={k:prior[k] for k in keys} for j,prior in zip(jobs,old['jobs'])),
            'Exactly the original ordered sixteen sample/video/prompt/rank identities required')


def prepare(model_proofs,output):
    auth=pinned(AUTH,AUTH_SHA256);recovery_auth=pinned(RECOVERY_AUTH,RECOVERY_AUTH_SHA256)
    validate_authorizations(auth,recovery_auth);history=freeze_history()
    require(not LOCAL.exists(),'Recovery local root already exists; never reset/reuse')
    require(set(model_proofs)==set(range(4)),'All four current model proofs required')
    nodes=[];identity=None
    for rank,path in sorted(model_proofs.items()):
        bound=record(path);proof=read(bound)
        require(proof['schema']=='teacher_label_model_content_v1' and proof['rank']==rank and proof['GPU_used'] is False
                and proof['weight_shards']==5 and proof['model_type']=='qwen2_5_vl','Actual current Qwen file proof required')
        validate_model_proof(proof,rank)
        current=canonical_models(proof);require(identity is None or current==identity,'Four-node model content differs');identity=current
        nodes.append(dict(rank=rank,hostname=proof['hostname'],proof=bound,model_root=proof['model_root'],model_files=proof['models']))
    require(len({n['hostname'] for n in nodes})==4,'Four distinct physical node identities required')
    manifest,rows=prep.manifest_rows(MANIFEST)
    eligible=[r for r in rows if not prep.UNSAFE.search(' '.join([r['sample_id'],Path(r['video']).name,*r['captions'].values()]))]
    require(len(rows)==1856 and len(eligible)==1815 and len(rows)-len(eligible)==41,'Original metadata pool/prescreen counts required')
    import random
    selected=random.Random(prep.SEED).sample(eligible,16)
    jobs=[]
    for index,row in enumerate(selected):
        sid=row['sample_id'];require(sid not in ('.','..') and all(c.isalnum() or c in '_.+-' for c in sid),'Unsafe sample basename')
        video=record(row['video']);jobs.append(dict(index=index,rank=index//4,sample_id=sid,video=video,
            original_prompt_sha256=hashlib.sha256(row['captions']['orig'].encode()).hexdigest(),
            local_video=str(LOCAL/f'video{index}.mp4')))
    validate_same_jobs(jobs,read(history['original_attempt2_plan']))
    require(manifest==read(history['original_attempt2_plan'])['manifest'],'Original manifest bytes required')
    value=dict(schema=SCHEMA,private=True,prompt_version=caption.PROMPT_VERSION,authorization=auth,recovery_authorization=recovery_auth,manifest=manifest,
        source_files=[record(ROOT/n,nonempty=(ROOT/n).stat().st_size>0) for n in SOURCE_NAMES],runtime=runtime(),nodes=nodes,jobs=jobs,
        allocations=[dict(rank=r,gpu_index=0,allocated_gpus=1,cap_seconds=CAP,job_indices=list(range(r*4,r*4+4))) for r in range(4)],
        stage='label_rebuild_and_cache',stage_cap_seconds=86400,pilot_reserved_seconds=3600,
        attempt_id=ATTEMPT_ID,local_root=str(LOCAL),**history,new_reserved_seconds=4*CAP,
        recovery_attempts_allowed=1,full_label_rebuild_allowed=False,AR_allowed=False,student_distillation_allowed=False,
        metadata_eligible_rows=len(eligible),metadata_rejected_rows=len(rows)-len(eligible),
        schema_validation_is_semantic_review=False,independent_semantic_review_pass=False,formal_training_allowed=False,
        retry_allowed=False,GPU_execution_started=False,selection_seed=prep.SEED,safety_filter_before_selection=True)
    return write_new(output,value)


def verify(path,digest,rank,local_assets=True):
    row=record(path);require(row['sha256']==digest,'Explicit approved pilot SHA required');p=read(row)
    require(p['schema']==SCHEMA and p['private'] is True and p['prompt_version']==caption.PROMPT_VERSION
            and p['stage_cap_seconds']==86400 and p['pilot_reserved_seconds']==3600
            and p['retry_allowed'] is False and p['independent_semantic_review_pass'] is False
            and p['schema_validation_is_semantic_review'] is False and p['formal_training_allowed'] is False
            and p['stage']=='label_rebuild_and_cache' and p['GPU_execution_started'] is False
            and p['full_label_rebuild_allowed'] is False and p['AR_allowed'] is False and p['student_distillation_allowed'] is False
            and type(p['recovery_attempts_allowed']) is int and p['recovery_attempts_allowed']==1
            and p['selection_seed']==20261101 and p['safety_filter_before_selection'] is True
            and p['metadata_eligible_rows']==1815 and p['metadata_rejected_rows']==41,'Pilot stage/schema drift')
    validate_authorizations(p['authorization'],p['recovery_authorization']);history=freeze_history()
    require(all(p[k]==v for k,v in history.items()) and p['attempt_id']==ATTEMPT_ID and p['local_root']==str(LOCAL)
            and p['new_reserved_seconds']==4*CAP and p['prior_attempt_charged_seconds']+p['new_reserved_seconds']<=3600,
            'Both immutable prior attempts or fresh recovery reservation drift')
    validate_same_jobs(p['jobs'],read(history['original_attempt2_plan']))
    require(type(rank) is int and rank in range(4) and len(p['jobs'])==16 and len(p['nodes'])==4,'Fixed pilot topology required')
    expected=[dict(rank=r,gpu_index=0,allocated_gpus=1,cap_seconds=CAP,job_indices=list(range(r*4,r*4+4))) for r in range(4)]
    require(len({j['sample_id'] for j in p['jobs']})==16
            and all(j['local_video']==str(LOCAL/f"video{j['index']}.mp4") for j in p['jobs']), 'Pilot media identities/path drift')
    require(p['allocations']==expected and [j['index'] for j in p['jobs']]==list(range(16))
            and [j['rank'] for j in p['jobs']]==[i//4 for i in range(16)],'Pilot all-four budget/job mapping drift')
    check(p['authorization']);check(p['manifest'])
    require(p['manifest']==read(history['original_attempt2_plan'])['manifest'],'Original manifest binding drift')
    require({r['path'] for r in p['source_files']}=={str(ROOT/n) for n in SOURCE_NAMES},'Source set drift')
    require(len(p['source_files'])==len(SOURCE_NAMES),'Exact current source coverage required')
    for source in p['source_files']:check(source)
    identities=[]
    for expected_rank,n in enumerate(p['nodes']):
        check(n['proof']);proof=validate_model_proof(read(n['proof']),expected_rank)
        require(n['rank']==expected_rank and n['hostname']==proof['hostname'] and n['model_root']==proof['model_root']
                and n['model_files']==proof['models'],'All four current model proof bindings required')
        identities.append(canonical_models(proof))
    require(all(v==identities[0] for v in identities),'Four-node model content differs')
    node=p['nodes'][rank];require(node['rank']==rank and node['hostname']==os.uname().nodename,'Wrong pilot physical host')
    check(node['proof']);proof=read(node['proof'])
    require(proof['schema']=='teacher_label_model_content_v1' and proof['rank']==rank and proof['hostname']==node['hostname']
            and proof['GPU_used'] is False and proof['weight_shards']==5 and proof['model_type']=='qwen2_5_vl'
            and node['model_root']==proof['model_root']
            and node['model_files']==proof['models'],'Model proof metadata/root drift')
    validate_model_proof(proof,rank)
    require([n['rank'] for n in p['nodes']]==list(range(4)) and len({n['hostname'] for n in p['nodes']})==4, 'Node proof sequence/identity drift')
    if local_assets:
        for model in node['model_files']:check({k:model[k] for k in ('path','bytes','sha256')})
        current=runtime();require(current['packages']==p['runtime']['packages'] and current['files']==p['runtime']['files']
                               and current['tools']==p['runtime']['tools'],'Local label runtime content drift')
    return p


def stage(path,digest,rank):
    p=verify(path,digest,rank)
    require(not LOCAL.exists(),'Old local pilot root exists; never reset/reuse')
    LOCAL.mkdir(mode=0o700)
    for job in p['jobs'][rank*4:rank*4+4]:
        check(job['video']);destination=Path(job['local_video'])
        with Path(job['video']['path']).open('rb') as src,destination.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
        require(record(destination)['sha256']==job['video']['sha256'],'Source/local media bytes differ')
    return write_new(LOCAL/'stage.json',dict(plan=record(path),rank=rank,hostname=os.uname().nodename,
         jobs=p['jobs'][rank*4:rank*4+4],GPU_used=False))


def verify_worker_ancestry(launch,p,rank,launch_path,state_dir,ancestors):
    require(Path(launch_path).absolute()==LOCAL/'launch.json' and launch['rank']==rank and launch['cap_seconds']==CAP
            and launch['state_dir']==str(Path(state_dir).absolute()) and launch['gpu_index']==0,
            'Worker canonical launch/rank/GPU lease drift')
    require(type(launch['start_monotonic']) in (int,float) and launch['start_monotonic']>0
            and type(launch['start_epoch']) in (int,float) and launch['start_epoch']>0,'Invalid allocation clocks')
    parent=launch['parent']
    require(any(all(a[k]==parent[k] for k in ('pid','session','start_ticks','argv')) for a in ancestors),
            'Actual worker ancestry does not contain the approved parent')
    leases=[]
    for ancestor in ancestors:
        args=ancestor['argv']
        prefix=args[:args.index('--')] if '--' in args else args
        if str(ROOT/'scripts/gpu_hold.py') in prefix:
            leases.append(dict(ancestor,argv=prefix))
    require(leases,'Actual GPU exec ancestry missing')
    required=(('--gpus','0'),('--state-dir',str(Path(state_dir).absolute())),('--action','exec'))
    require(any(all(a['argv'].count(flag)==1 and a['argv'].index(flag)+1<len(a['argv'])
                    and a['argv'][a['argv'].index(flag)+1]==value for flag,value in required) for a in leases),
            'Wrong actual GPU exec lease/GPU/state')
    return True


def worker(path,digest,rank,launch_path,state_dir):
    launch=read(record(launch_path));start=launch['start_monotonic']
    def budget_check():require(time.monotonic()-start<CAP-TAIL-10,'Pilot allocated wall exhausted')
    budget_check();p=verify(path,digest,rank)
    require(launch['plan']==record(path) and launch['rank']==rank and process.same_process(launch['parent']),'Worker parent/plan binding drift')
    parent=process.identity(launch['parent']['pid']);args=parent['argv']
    require(str(Path(__file__).resolve()) in args and 'run' in args and '--gpu-authorized' in args,'Only the controlled parent can start CUDA')
    for flag,value in (('--plan',str(Path(path).absolute())),('--approved-plan-sha256',digest),('--rank',str(rank))):
        require(args.count(flag)==1 and args[args.index(flag)+1]==value,'Parent task mismatch')
    ancestors=[];pid=os.getppid()
    while pid>1:
        identity=process.identity(pid);ancestors.append(identity);pid=identity['ppid']
    require(any(Path(a['argv'][0]).name=='timeout' and '--signal=TERM' in a['argv'] and '--kill-after=20s' in a['argv'] and str(CAP-TAIL)+'s' in a['argv'] for a in ancestors)
            and any('gpu_hold.py' in {Path(x).name for x in a['argv']} and 'exec' in a['argv'] for a in ancestors), 'GNU timeout and GPU lease ancestry required')
    verify_worker_ancestry(launch,p,rank,launch_path,state_dir,ancestors)
    supervised_path=LOCAL/'supervised.json'
    while not supervised_path.is_file():
        budget_check();require(process.same_process(launch['parent']),'Parent vanished before owned dispatch publication');time.sleep(.01)
    supervised=read(record(supervised_path));child=supervised['child']
    require(supervised['launch']==record(launch_path) and child['pid']==child['session']
            and child['ppid']==launch['parent']['pid']
            and any(a['pid']==child['pid'] and a['start_ticks']==child['start_ticks'] for a in ancestors),
            'Actual fresh child session/parent launch binding missing')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==launch['gpu_uuid'],'Exactly the approved GPU UUID required')
    boundary=process.manager_boundary(state_dir,0,launch['gpu_uuid'],require_idle=False)
    require(set(boundary['query']['pids']) <= {a['pid'] for a in ancestors} | {os.getpid()},
            'External GPU workload appeared before CUDA; do not overlap it')
    check(p['source_files'][0]);budget_check()
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    require(torch.cuda.device_count()==1,'Exactly one visible GPU')
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(p['nodes'][rank]['model_root'],torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    processor=AutoProcessor.from_pretrained(p['nodes'][rank]['model_root'],local_files_only=True)
    records=[]
    for job in p['jobs'][rank*4:rank*4+4]:
        budget_check();local=Path(job['local_video']);require(record(local)['sha256']==job['video']['sha256'],'Local media drift before recaption')
        started=time.monotonic()
        try:
            original_sampler=caption.sample_frames
            def exact_sample(*args,**kwargs):
                frames=original_sampler(*args,**kwargs)
                require(len(frames)==caption.N_FRAMES,'Exact eight source frames required')
                return frames
            caption.sample_frames=exact_sample
            try:
                facts,caption_text,removed=caption.recaption_one(model,processor,str(local),model.device)
            finally:
                caption.sample_frames=original_sampler
            caption.validate_facts(facts)
            require(not prep.UNSAFE.search(json.dumps(facts,ensure_ascii=False)), 'Sensitive model facts rejected without publication')
            require(caption_text.strip() and not prep.UNSAFE.search(caption_text),'Unsafe/empty model caption rejected')
            item=dict(sample_id=job['sample_id'],video_sha256=job['video']['sha256'],original_prompt_sha256=job['original_prompt_sha256'],
                prompt_version=caption.PROMPT_VERSION,facts_json=facts,caption=caption_text,removed_sentences=removed,
                frame_indices=list(range(caption.N_FRAMES)),model='Qwen2.5-VL-7B-Instruct',semantic_review_pass=False)
        except Exception as exc:item=dict(sample_id=job['sample_id'],prompt_version=caption.PROMPT_VERSION,error_type=type(exc).__name__,semantic_review_pass=False)
        item['elapsed_seconds']=time.monotonic()-started
        write_new(LOCAL/f'label{job["index"]}.json',item);records.append(item);budget_check()
    return write_new(LOCAL/'worker.json',dict(plan=record(path),rank=rank,success=True,records=records,
        end_monotonic=time.monotonic(),end_epoch=time.time(),semantic_review_pass=False,formal_training_allowed=False))


def run(path,digest,rank,state_dir,gpu_authorized=False):
    require(gpu_authorized is True,'Explicit GPU pilot authorization required')
    p=verify(path,digest,rank);stage_path=LOCAL/'stage.json';s=read(record(stage_path));require(s['plan']==record(path) and s['rank']==rank,'Actual local media stage required')
    require(not (LOCAL/'launch.json').exists() and not (LOCAL/'parent.json').exists(),'Pilot already attempted; no retry')
    boundary=process.manager_boundary(state_dir,0);owner=process.identity(os.getpid());start,epoch=time.monotonic(),time.time()
    launch=dict(plan=record(path),rank=rank,parent=owner,gpu_uuid=boundary['gpu_uuid'],start_monotonic=start,start_epoch=epoch,cap_seconds=CAP,gpu_index=0,state_dir=str(Path(state_dir).absolute()))
    write_new(LOCAL/'launch.json',launch)
    command=[sys.executable,str(ROOT/'scripts/gpu_hold.py'),'--action','exec','--gpus','0','--state-dir',str(state_dir),'--',
        p['runtime']['tools']['timeout'],'--signal=TERM','--kill-after=20s',str(CAP-TAIL)+'s',sys.executable,'-B',str(Path(__file__).resolve()),'worker',
        '--plan',str(path),'--approved-plan-sha256',digest,'--rank',str(rank),'--state-dir',str(state_dir),'--launch',str(LOCAL/'launch.json')]
    child=None;identity=None;tracked=[];error=None;released=False;queries=[];old={}
    interrupted=[]
    def stop(signum,_frame):interrupted.append(signum)  # Repeated signals must not interrupt owned cleanup or receipt saving.
    for sig in (signal.SIGTERM,signal.SIGINT):old[sig]=signal.getsignal(sig);signal.signal(sig,stop)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=boundary['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
        with (LOCAL/'worker.log').open('xb') as log:child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process.identity(child.pid)
        require(identity['session']==identity['pid'] and identity['ppid']==owner['pid'],'Fresh owned child session required')
        write_new(LOCAL/'supervised.json',dict(launch=record(LOCAL/'launch.json'),child=identity,command=command))
        while child.poll() is None and not interrupted and time.monotonic()<start+CAP-TAIL:process.owned_members(identity,tracked);time.sleep(.1)
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        try:
            if child and identity:process.cleanup_owned(child,identity,start+CAP-50,tracked)
            elif child and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
            if child and identity:
                queries=[process.query_release(boundary['gpu_uuid']) for _ in range(2)]
                released=not process.owned_members(identity,tracked) and all(q['pids']==[] for q in queries)
        except BaseException as exc:error='Cleanup/release '+type(exc).__name__+': '+str(exc)
        worker_path=LOCAL/'worker.json'
        worker_value=None
        if worker_path.is_file():
            try:worker_value=read(record(worker_path))
            except (ValueError,OSError,KeyError):pass
        elapsed=time.monotonic()-start
        result=dict(schema='teacher_label_pilot_parent_v1',plan=record(path),rank=rank,parent=owner,child=identity,
            command=command,start_epoch=epoch,end_epoch=time.time(),start_monotonic=start,end_monotonic=start+elapsed,
            elapsed_seconds=elapsed,cap_seconds=CAP,charged_upper_seconds=elapsed if released else max(CAP,elapsed),actual_elapsed_seconds=elapsed if released else None,
            released=released,release_queries=queries,success=False,error=error,exit_code=child.returncode if child else None,
            semantic_review_pass=False,formal_training_allowed=False)
        if worker_value is not None:
            result['worker']=record(worker_path)
            result['success']=released and child.returncode==0 and worker_value['success'] is True and len(worker_value['records'])==4 and all('error_type' not in x for x in worker_value['records']) and error is None and not interrupted and elapsed<=CAP
        result['interrupted_signals']=interrupted
        try:
            write_new(LOCAL/'parent.json',result)
            final_elapsed=time.monotonic()-start
            write_new(LOCAL/'cost.json',dict(plan=record(path),rank=rank,elapsed_seconds=final_elapsed,
                charged_upper_seconds=final_elapsed if released else max(CAP,final_elapsed),
                release_verified=released,budget_exceeded=final_elapsed>CAP,scope='pre_exec_through_final_release_and_parent_save'))
        finally:
            for sig,value in old.items():signal.signal(sig,value)
    require(result['success'],'Pilot failed; costs and labels preserved, no retry')
    return record(LOCAL/'parent.json')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','stage','run','worker'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--approved-plan-sha256');p.add_argument('--model-proofs',type=Path)
    p.add_argument('--rank',type=int);p.add_argument('--state-dir',type=Path);p.add_argument('--launch',type=Path);p.add_argument('--gpu-authorized',action='store_true');a=p.parse_args(argv);os.umask(0o077)
    if a.action=='prepare':
        require(not a.gpu_authorized and a.model_proofs,'CPU prepare requires modelproof list, not GPU flags')
        value=prep.strict_json(a.model_proofs.read_bytes())
        require(isinstance(value,dict) and set(value)=={'0','1','2','3'},'Exact four typed rank modelproof paths required')
        result=prepare({int(k):v for k,v in value.items()},a.plan)
    else:
        require(a.approved_plan_sha256 and a.rank is not None,'Approved plan SHA and fixed rank required')
        if a.action=='stage':require(not a.gpu_authorized,'CPU stage is not GPU authorization');result=stage(a.plan,a.approved_plan_sha256,a.rank)
        elif a.action=='run':result=run(a.plan,a.approved_plan_sha256,a.rank,a.state_dir,a.gpu_authorized)
        else:require(not a.gpu_authorized and a.launch and a.state_dir,'Worker only controlled parent');result=worker(a.plan,a.approved_plan_sha256,a.rank,a.launch,a.state_dir)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
