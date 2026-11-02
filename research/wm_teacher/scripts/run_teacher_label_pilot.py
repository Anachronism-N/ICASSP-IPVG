#!/usr/bin/env python3
"""One-shot, four-node label pilot, not a semantic or Teacher quality claim."""
import argparse
import hashlib
import importlib.metadata
import json
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
LOCAL=Path('/dockerdata/wm_tmp/teacher_label_pilot_20261102_attempt2')
ATTEMPT_ID='label_pilot_attempt2'
PRIOR_COST_SECONDS={0:18.143487762659788,1:15.434166906867176,2:19.951568322256207,3:17.300606979057193}
AUTH=ROOT/'eval/diag/teacher_training_admission_authorization_20261101.json'
MANIFEST=ROOT/'training_metadata/e3_recaption_20261022/pool1856_dual_manifest.jsonl'
SCHEMA='teacher_label_pilot_execution_v2'
CAP=860
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


def prepare(model_proofs,output):
    auth=record(AUTH);contract=read(auth)
    require(contract['stage_caps_GPUh']=={'label_rebuild_and_cache':24,'single_gpu_training_resume':2,'multi_node_technical_validation':6}
            and contract['total_cap_GPUh']==32 and contract['formal_candidate_training_budget_approved'] is False
            and contract['phase_borrowing_allowed'] is False,'Training admission authorization drift')
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
    require(len(eligible)>=16,'Not enough metadata-prescreened videos')
    import random
    selected=random.Random(prep.SEED).sample(eligible,16)
    prior_rows=[]
    for rank in range(4):
        bound=record(ROOT/f'eval/teacher_label_pilot_20261102/attempt1_rank{rank}.json')
        prior=read(bound)
        require(prior['cost']['release_verified'] is True, 'Previous allocation release missing')
        require(prior['parent']['released'] is True and prior['parent']['success'] is False
                and prior['cost']['charged_upper_seconds']==PRIOR_COST_SECONDS[rank]
                and prior['CUDA_initialized'] is False and prior['labels_written']==0, 'Prior failed pilot cost cannot be reset')
        prior_rows.append(bound)
    require(sum(PRIOR_COST_SECONDS.values())+4*CAP<=3600,'Repeated pilot plus previous cost exceeds original1GPUh pilot reservation')
    jobs=[]
    for index,row in enumerate(selected):
        sid=row['sample_id'];require(sid not in ('.','..') and all(c.isalnum() or c in '_.+-' for c in sid),'Unsafe sample basename')
        video=record(row['video']);jobs.append(dict(index=index,rank=index//4,sample_id=sid,video=video,
            original_prompt_sha256=hashlib.sha256(row['captions']['orig'].encode()).hexdigest(),
            local_video=str(LOCAL/f'video{index}.mp4')))
    value=dict(schema=SCHEMA,private=True,prompt_version=caption.PROMPT_VERSION,authorization=auth,manifest=manifest,
        source_files=[record(ROOT/n,nonempty=(ROOT/n).stat().st_size>0) for n in SOURCE_NAMES],runtime=runtime(),nodes=nodes,jobs=jobs,
        allocations=[dict(rank=r,gpu_index=0,allocated_gpus=1,cap_seconds=CAP,job_indices=list(range(r*4,r*4+4))) for r in range(4)],
        stage='label_rebuild_and_cache',stage_cap_seconds=86400,pilot_reserved_seconds=3600,
        attempt_id=ATTEMPT_ID,prior_attempt_proofs=prior_rows,prior_attempt_charged_seconds=sum(PRIOR_COST_SECONDS.values()),new_reserved_seconds=4*CAP,
        schema_validation_is_semantic_review=False,independent_semantic_review_pass=False,formal_training_allowed=False,
        retry_allowed=False,GPU_execution_started=False,selection_seed=prep.SEED,safety_filter_before_selection=True)
    return write_new(output,value)


def verify(path,digest,rank,local_assets=True):
    row=record(path);require(row['sha256']==digest,'Explicit approved pilot SHA required');p=read(row)
    require(p['schema']==SCHEMA and p['private'] is True and p['prompt_version']==caption.PROMPT_VERSION
            and p['stage_cap_seconds']==86400 and p['pilot_reserved_seconds']==3600
            and p['retry_allowed'] is False and p['independent_semantic_review_pass'] is False
            and p['formal_training_allowed'] is False,'Pilot stage/schema drift')
    require(p['attempt_id']==ATTEMPT_ID and len(p['prior_attempt_proofs'])==4
            and p['prior_attempt_charged_seconds']==sum(PRIOR_COST_SECONDS.values()) and p['new_reserved_seconds']==4*CAP
            and p['prior_attempt_charged_seconds']+p['new_reserved_seconds']<=3600, 'Prior failed pilot charges or remaining reservation drift')
    for prior_rank,proof in enumerate(p['prior_attempt_proofs']):
        prior=read(proof)
        require(prior['cost']['charged_upper_seconds']==PRIOR_COST_SECONDS[prior_rank]
                and prior['cost']['release_verified'] is True and prior['parent']['released'] is True
                and prior['CUDA_initialized'] is False and prior['labels_written']==0, 'Prior allocation evidence unverified')
    require(type(rank) is int and rank in range(4) and len(p['jobs'])==16 and len(p['nodes'])==4,'Fixed pilot topology required')
    expected=[dict(rank=r,gpu_index=0,allocated_gpus=1,cap_seconds=CAP,job_indices=list(range(r*4,r*4+4))) for r in range(4)]
    require(len({j['sample_id'] for j in p['jobs']})==16
            and all(j['local_video']==str(LOCAL/f"video{j['index']}.mp4") for j in p['jobs']), 'Pilot media identities/path drift')
    require(p['allocations']==expected and [j['index'] for j in p['jobs']]==list(range(16))
            and [j['rank'] for j in p['jobs']]==[i//4 for i in range(16)],'Pilot all-four budget/job mapping drift')
    check(p['authorization']);check(p['manifest'])
    authorization=read(p['authorization'])
    require(authorization['stage_caps_GPUh']=={'label_rebuild_and_cache':24,'single_gpu_training_resume':2,'multi_node_technical_validation':6}
            and authorization['total_cap_GPUh']==32 and authorization['phase_borrowing_allowed'] is False
            and authorization['formal_candidate_training_budget_approved'] is False, 'Actual stage authorization mismatch')
    require({r['path'] for r in p['source_files']}=={str(ROOT/n) for n in SOURCE_NAMES},'Source set drift')
    for source in p['source_files']:check(source)
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
    for flag,value in (('--gpus','0'),('--state-dir',str(Path(state_dir).absolute())),('--action','exec')):
        require(any(a['argv'].count(flag)==1 and a['argv'][a['argv'].index(flag)+1]==value for a in leases),
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
        value=json.loads(a.model_proofs.read_bytes());result=prepare({int(k):v for k,v in value.items()},a.plan)
    else:
        require(a.approved_plan_sha256 and a.rank is not None,'Approved plan SHA and fixed rank required')
        if a.action=='stage':require(not a.gpu_authorized,'CPU stage is not GPU authorization');result=stage(a.plan,a.approved_plan_sha256,a.rank)
        elif a.action=='run':result=run(a.plan,a.approved_plan_sha256,a.rank,a.state_dir,a.gpu_authorized)
        else:require(not a.gpu_authorized and a.launch and a.state_dir,'Worker only controlled parent');result=worker(a.plan,a.approved_plan_sha256,a.rank,a.launch,a.state_dir)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
