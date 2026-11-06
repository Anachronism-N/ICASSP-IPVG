#!/usr/bin/env python3
"""Finite native-video/JSON-caption pilot; not semantic or formal training approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'eval/teacher_progress_campaign_20261104/allocation004_temporal'
AUTH_SHA='e911d18de9140b202fc3e31e0ebefe447804f7dadf907e121c701a15caf0c0bb'
CAP=5400
SCHEMA_DECODE=False
BOUNDARY_DECODE=False
BOUNDED_DECODE=False
WORD_ACTOR_DECODE=False
CARDS=4
UPPER_GPUH=6
DECODER_SITE=Path('/dockerdata/wm_json_decoder_CPU_20261104/site')
STRUCTURED_INSTANCE=None
PYTHON='/dockerdata/venv_qwen32b_20261103/bin/python3.12'
MODEL='/opt/models/Qwen3-VL-32B-Instruct'
CAPTION_PROMPT="""Compare the visible temporal sequence and the grounded facts below. Return ONLY a JSON
array of exactly four strings, each a complete declarative sentence of 12 to 22 English words.
Together the four sentences must be 48 to 88 words. Describe visible entities, their supported
changes and the final visible state. Associate each action with the correct entity. Do not infer
motion solely from a pose, and do not erase visible motion merely because endpoints match.
Do not invent actions to fill the word count. Do not state identity, intent, unreadable text, or
input metadata. Do not output numbered lists, explanations or markdown. Grounded facts:
"""

def configure_schema():
    global E,AUTH_SHA,SCHEMA_DECODE
    E=ROOT/'eval/teacher_progress_campaign_20261104/allocation005_schema'
    AUTH_SHA='f57076a67eeabe23b429d273b2c8b988ca847958fc694657e07cb1401576c7b2'
    SCHEMA_DECODE=True
    sys.path.insert(0,str(DECODER_SITE))


def configure_boundary():
    global E,AUTH_SHA,BOUNDARY_DECODE
    configure_schema()
    E=ROOT/'eval/teacher_progress_campaign_20261104/allocation006_boundary'
    AUTH_SHA='ea579ae76a13ff28675bb25d25d59664199a865dab59caba802707da0cfa3646'
    BOUNDARY_DECODE=True


def configure_bounded():
    global E,AUTH_SHA,BOUNDED_DECODE,CARDS,UPPER_GPUH
    configure_boundary()
    E=ROOT/'eval/teacher_progress_campaign_20261104/allocation007_bounded'
    AUTH_SHA='312240ef08313c754ccc6c1ff313eaa1dae435e18049ff78496958d5609121c1'
    BOUNDED_DECODE=True
    CARDS=2
    UPPER_GPUH=3


ACTOR_PROMPT = """
For every event, use its entity_id as the sole grammatical actor. Describe that entity's
visible change over consecutive observations, not a different entity's hands or held object.
When two entities act, use separate events with the corresponding IDs. Keep locations and
held objects in the relevant entity record. For interaction, describe only the current
actor's own visible action, never transfer another actor's action into this event.
Compare early, middle and late observations before naming an action: visible evidence,
not a single pose, must support it. Preserve return motion and supported final state.
These instructions do not justify inventing an event or suppressing actual visible motion.
"""


def configure_word_actor():
    global E,AUTH_SHA,WORD_ACTOR_DECODE
    configure_bounded()
    E=ROOT/'eval/teacher_progress_campaign_20261104/allocation008_word_actor'
    AUTH_SHA='74770bf4f5bcbba1229f26fd8a81d75173fb2bfb7aff93c351514d54334da7e3'
    WORD_ACTOR_DECODE=True


def facts_prompt():
    r,_=helpers()
    return r.FACTS_PROMPT + (ACTOR_PROMPT if WORD_ACTOR_DECODE else '')


def helpers():
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_teacher_qwen3vl32b_recovery as r
    import teacher_gpu_exclusive as g
    return r,g

def record(path):
    return helpers()[0].record(path)

def read(row):
    return helpers()[0].read(row)

def save(path,value):
    return helpers()[0].write_new(path,value)

def authorization():
    row=record(E/('authorization.final.twocard.json' if WORD_ACTOR_DECODE else 'authorization.twocard.json' if BOUNDED_DECODE else 'authorization.final.json' if BOUNDARY_DECODE else 'authorization.json'));assert row['sha256']==AUTH_SHA
    a=read(row)
    assert record(E.parent/'campaign.authorization.json')['sha256']==a['parent_campaign_sha256']
    assert a['allocated_cards']==CARDS and a['wall_cap_seconds']==CAP and a['upper_GPUh']==UPPER_GPUH
    assert a['prior_new_label_GPUh']==(43 if WORD_ACTOR_DECODE else 39 if BOUNDED_DECODE else 33 if BOUNDARY_DECODE else 27 if SCHEMA_DECODE else 20)
    assert a['prior_historical_label_GPUh']==(66.832564877520134 if WORD_ACTOR_DECODE else 62.832564877520134 if BOUNDED_DECODE else 56.832564877520134 if BOUNDARY_DECODE else 50.832564877520134 if SCHEMA_DECODE else 43.832564877520134)
    assert a['prior_new_label_GPUh']+UPPER_GPUH<=a['stage_label_cap_GPUh']==96
    assert a['cases']==32 and a['source_frames']==16 and a['source_indices']==list(range(0,121,8))
    assert a['caption_words']==[40,120] and a['automatic_retry_allowed'] is False
    assert not a['formal_training_allowed'] and not a['semantic_approval']
    if SCHEMA_DECODE:
        previous=E.parent/('allocation007_bounded' if WORD_ACTOR_DECODE else 'allocation006_boundary' if BOUNDED_DECODE else 'allocation005_schema' if BOUNDARY_DECODE else 'allocation004_temporal')
        assert record(previous/('completion.independent_audit.v2.json' if BOUNDED_DECODE and not WORD_ACTOR_DECODE else 'completion.independent_audit.json'))['sha256']==a['previous_native_audit_sha256']
        assert record(previous/'run01/attempt.claim')['sha256']==a['previous_claim_sha256']
        assert record(E.parent/('semantic_review007/completion.independent_audit.json' if WORD_ACTOR_DECODE else 'semantic_review005/completion.independent_audit.json'))['sha256']==a['previous_review_sha256']
        probe=record(E/'actual_CPU_probe.final.json' if BOUNDARY_DECODE else DECODER_SITE.parent/'actual_CPU_probe.json')
        assert probe['sha256']==a['decoder_CPU_probe_sha256']
        if BOUNDARY_DECODE:
            proof=read(probe);assert proof['success'] and not proof['CUDA_initialized']
            assert proof['source_sha256']==record(ROOT/'scripts/teacher_structured_decoder.py')['sha256']
            prior=read(record(previous/('completion.independent_audit.v2.json' if BOUNDED_DECODE and not WORD_ACTOR_DECODE else 'completion.independent_audit.json')))
            assert prior['cases_attempted']==(1 if BOUNDED_DECODE and not WORD_ACTOR_DECODE else 32) and prior['labels']==(30 if WORD_ACTOR_DECODE else 0) and prior['released'] and prior['modes_restored']
            if WORD_ACTOR_DECODE:
                assert proof['caption_word_count_path_pass'] and proof['word_count_boundary_mask_pass']
                assert a['caption_schema']==config()['caption_schema']
                assert a['actor_prompt_sha256']==hashlib.sha256(ACTOR_PROMPT.encode()).hexdigest()
                reviewed=read(record(E.parent/'semantic_review007/completion.independent_audit.json'))
                assert reviewed['success'] and reviewed['reviewed_labels']==30
                assert reviewed['adequate_over_original32']==21 and reviewed['released'] and reviewed['modes_restored']
    return a,row

def config():
    r,_=helpers();r.configure_prompt_v2()
    value=dict(model=MODEL,dtype='bfloat16',cards=CARDS,do_sample=False,source_frames=16,
        source_fps=24,source_indices=list(range(0,121,8)),resize=[640,352],
        do_sample_frames=False,temporal_patch_size=2,facts_max_tokens=1024,caption_max_tokens=384,
        caption_words=[40,120],facts_prompt_sha256=hashlib.sha256(facts_prompt().encode()).hexdigest(),
        caption_prompt_sha256=hashlib.sha256(CAPTION_PROMPT.encode()).hexdigest(),
        caption_render='exact_json_array4_join_single_spaces_no_edit',formal_training_allowed=False)
    if SCHEMA_DECODE:
        import teacher_structured_decoder as structured
        value['decoding']='JSON_schema_and_original_deny_vocabulary'
        value['facts_schema']=structured.FACTS_SCHEMA;value['caption_schema']=structured.CAPTION_SCHEMA
        value['denied_facts_words']=list(r.DENY_DECODE_WORDS);value['denied_caption_words']=list(r.DENY_DECODE_WORDS+r.CAPTION_DECODE_WORDS)
    if BOUNDARY_DECODE:
        value['decoder_boundary_tokens']='character_validated_quote_crossing_trie_v1'
        value['stop_on_generation_without_EOS_or_AssertionError']=True
    if BOUNDED_DECODE:
        value['facts_schema']=structured.bounded_facts_schema()
        value['bounded_facts_strings']=True
    if WORD_ACTOR_DECODE:
        value['caption_schema']=structured.caption_word_schema()
        value['caption_word_count_grammar']=True
        value['entity_specific_action_prompt']=True
    return value

def prepare(output=None):
    r,g=helpers();a,auth=authorization()
    original=record(E.parent/('allocation007_bounded/plan.final.json' if WORD_ACTOR_DECODE else 'allocation006_boundary/plan.final.json' if BOUNDED_DECODE else 'allocation005_schema/plan.ready.json' if BOUNDARY_DECODE else 'allocation004_temporal/plan.ready.json' if SCHEMA_DECODE else 'allocation003/plan.execution003.ready.json'))
    assert original['sha256']==a['original_plan_sha256'];p=read(original)
    audit=record(E.parent/('allocation007_bounded/completion.independent_audit.json' if WORD_ACTOR_DECODE else 'allocation006_boundary/completion.independent_audit.v2.json' if BOUNDED_DECODE else 'allocation005_schema/completion.independent_audit.json' if BOUNDARY_DECODE else 'allocation004_temporal/completion.independent_audit.json' if SCHEMA_DECODE else 'allocation003/completion.audit.json'))
    review=record(E.parent/('semantic_review007/completion.independent_audit.json' if WORD_ACTOR_DECODE else 'semantic_review005/completion.independent_audit.json' if SCHEMA_DECODE else 'semantic_review004/completion.independent_audit.json'))
    if not SCHEMA_DECODE:
        assert audit['sha256']==a['original_audit_sha256'] and review['sha256']==a['previous_review_sha256']
    assert read(audit)['cases_attempted']==(1 if BOUNDED_DECODE and not WORD_ACTOR_DECODE else 32) and read(review)['all_cases_denominator']==32
    jobs=[dict(index=j['index'],group=j['group'],video=j['video']) for j in p['jobs']]
    assert [j['index'] for j in jobs]==list(range(32))
    for j in jobs:r.check(j['video'])
    for m in p['model']['models']:r.check({k:m[k] for k in ('path','bytes','sha256')})
    names=('scripts/run_teacher_temporal_labels.py','tests/test_teacher_temporal_labels.py',*r.SOURCE_NAMES)
    if SCHEMA_DECODE:names+=('scripts/teacher_structured_decoder.py','tests/test_teacher_structured_decoder.py')
    plan=dict(schema='teacher_temporal_json_plan_v1',authorization=auth,original=original,
        original_audit=audit,previous_review=review,jobs=jobs,model=p['model'],runtime=r.runtime(),
        source={n:record(ROOT/n) for n in names},gpu_uuids=p['gpu_uuids'][-2:] if BOUNDED_DECODE else p['gpu_uuids'],hostname=os.uname().nodename,
        resource_control_runtime=g.control_runtime(),config=config(),output_root=a['output_root'],
        automatic_retry_allowed=False,semantic_approval=False,formal_training_allowed=False)
    if SCHEMA_DECODE:
        plan['decoder_runtime_files']=[record(f) for f in sorted(DECODER_SITE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith('.pyc')]
    return save(Path(output) if output else E/'plan.json',plan)

def verify(path,digest):
    r,g=helpers();a,auth=authorization();bound=record(path);assert bound['sha256']==digest;p=read(bound)
    assert p['authorization']==auth and p['config']==config() and p['hostname']==os.uname().nodename
    assert p['resource_control_runtime']==g.control_runtime() and p['runtime']==r.runtime()
    assert p['output_root']==a['output_root'] and p['automatic_retry_allowed'] is False
    assert p['semantic_approval'] is False and p['formal_training_allowed'] is False
    assert [j['index'] for j in p['jobs']]==list(range(32))
    assert [j['group'] for j in p['jobs']]==['diagnostic']*16+['blind']*16
    original=read(p['original'])
    assert p['jobs']==[dict(index=j['index'],group=j['group'],video=j['video']) for j in original['jobs']]
    assert p['model']==original['model'] and p['gpu_uuids']==(original['gpu_uuids'][-2:] if BOUNDED_DECODE else original['gpu_uuids'])
    for row in [p['original'],p['original_audit'],p['previous_review'],*p['source'].values()]:r.check(row)
    for m in p['model']['models']:r.check({k:m[k] for k in ('path','bytes','sha256')})
    for j in p['jobs']:r.check(j['video'])
    assert len(p['gpu_uuids'])==len(set(p['gpu_uuids']))==CARDS
    if SCHEMA_DECODE:
        assert p['decoder_runtime_files']
        for row in p['decoder_runtime_files']:r.check(row)
    return p,bound

def native_inputs(processor,images,prompt):
    import numpy as np
    assert len(images)==16 and all(im.size==(640,352) for im in images)
    messages=[dict(role='user',content=[dict(type='video'),dict(type='text',text=prompt)])]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    metadata=dict(fps=24,total_num_frames=121,frames_indices=list(range(0,121,8)),
                  width=1280,height=704,duration=121/24)
    inputs=processor(text=[text],videos=[np.stack([np.asarray(im) for im in images])],
        video_metadata=[metadata],do_sample_frames=False,return_tensors='pt')
    assert inputs['video_grid_thw'].shape[0]==1
    assert int(inputs['video_grid_thw'][0,0])*processor.video_processor.temporal_patch_size==16
    assert 'image_grid_thw' not in inputs
    return inputs

def parse_caption(text):
    r,_=helpers()
    value=json.loads(text)
    assert type(value) is list and len(value)==4 and all(type(s) is str and s==s.strip() and s and '\n' not in s for s in value)
    return r.validate_caption(' '.join(value))

def generate(model,processor,images,prompt,limit,trace):
    import torch
    inputs=native_inputs(processor,images,prompt).to(model.get_input_embeddings().weight.device)
    constraints=STRUCTURED_INSTANCE.constraints(facts_stage=limit==1024) if SCHEMA_DECODE else {}
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=limit,do_sample=False,**constraints)
    tokens=out[:,inputs['input_ids'].shape[1]:];trace['new_tokens']=int(tokens.shape[1])
    eos=model.generation_config.eos_token_id;eos=[eos] if type(eos) is int else eos
    if BOUNDARY_DECODE:
        trace['ended_with_EOS']=bool(tokens.shape[1] and tokens[0,-1].item() in eos)
        sent=out[0,:-1] if trace['ended_with_EOS'] else out[0]
        fn=constraints['prefix_allowed_tokens_fn'];allowed=fn(0,sent)
        state=fn.token_enforcer.prefix_states[tuple(sent.tolist())].parser
        trace['grammar_can_end']=state.can_end()
        trace['grammar_EOS_allowed']=processor.tokenizer.eos_token_id in allowed
        trace['grammar_allowed_token_count']=len(allowed)
        trace['grammar_inside_freetext']=bool(state.shortcut_key())
    assert tokens.shape[0]==1 and tokens.shape[1]>0 and tokens[0,-1].item() in eos
    return processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

def worker(args):
    global STRUCTURED_INSTANCE
    r,g=helpers();p,bound=verify(args.plan,args.sha);launch=read(record(E/'run01/launch.json'))
    process=r.helpers()[1];assert process.same_process(launch['parent'])
    assert os.getppid()==launch['parent']['pid'] and os.environ['CUDA_VISIBLE_DEVICES']==','.join(p['gpu_uuids'])
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    assert torch.cuda.device_count()==CARDS
    started=time.monotonic();end=launch['started_monotonic']+CAP-120;rows=[];worst=0.;stop='completed'
    if SCHEMA_DECODE:
        from teacher_structured_decoder import StructuredDecoder
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        eos=json.loads(Path(MODEL,'generation_config.json').read_bytes())['eos_token_id']
        STRUCTURED_INSTANCE=StructuredDecoder(processor.tokenizer,eos,bounded=BOUNDED_DECODE,word_count=WORD_ACTOR_DECODE)
        save(E/'run01/decoder_cuda_admission.json',STRUCTURED_INSTANCE.cuda_admission('cuda:'+str(CARDS-1)))
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='balanced',max_memory={i:('70GiB' if BOUNDED_DECODE else '34GiB') for i in range(CARDS)},attn_implementation='sdpa')
    if BOUNDED_DECODE:assert set(model.hf_device_map.values())==set(range(CARDS))
    else:r.validate_device_map(model.hf_device_map)
    assert {v.device.index for v in model.parameters()}==set(range(CARDS))
    assert all(v.device.type=='cuda' and v.dtype==torch.bfloat16 for v in model.parameters())
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);load_seconds=time.monotonic()-started
    decoder_stop=False
    for j in p['jobs']:
        if end-time.monotonic()<=worst*(32-j['index'])+10:stop='projected_budget_insufficient';break
        began=time.monotonic();stage='frame_decode';trace={};counts={}
        try:
            r.check(j['video'],end);images,frames=r.sample_images(j['video']['path'],p['runtime']['tools'],end)
            assert frames['source_indices']==list(range(0,121,8))
            stage='facts_generate';trace={};raw=generate(model,processor,images,facts_prompt(),1024,trace)
            counts['facts']=trace['new_tokens'];stage='facts_parse';facts=r.parse_facts(raw)
            stage='caption_generate';trace={};raw=generate(model,processor,images,CAPTION_PROMPT+json.dumps(facts,sort_keys=True),384,trace)
            counts['caption']=trace['new_tokens'];stage='caption_validate';caption=parse_caption(raw)
            value=dict(index=j['index'],group=j['group'],video=j['video'],facts_json=facts,caption=caption,
                       success=True,duration_seconds=time.monotonic()-began,new_tokens=counts)
        except Exception as exc:
            if trace.get('new_tokens') is not None and stage.endswith('generate'):counts['facts' if stage=='facts_generate' else 'caption']=trace['new_tokens']
            value=dict(index=j['index'],group=j['group'],video=j['video'],success=False,
                       error_type=type(exc).__name__,failure_stage=stage,duration_seconds=time.monotonic()-began,new_tokens=counts)
            if SCHEMA_DECODE:value.update(r.case_failure(exc,stage,time.monotonic()-began,counts))
            if BOUNDARY_DECODE:
                value['decoder_diagnostic']=dict(trace)
                decoder_stop=trace.get('ended_with_EOS') is False or isinstance(exc,AssertionError)
        raw=None;row=save(E/'run01'/('label'+str(j['index'])+'.json'),value);rows.append(row)
        worst=max(worst,time.monotonic()-began)
        print(json.dumps(dict(case=j['index'],success=value['success'],stage=None if value['success'] else stage)),flush=True)
        if decoder_stop:stop='decoder_admission_failed_stop_without_retry';break
    attempted=len(rows)
    for j in p['jobs'][attempted:]:
        rows.append(save(E/'run01'/('label'+str(j['index'])+'.json'),dict(index=j['index'],group=j['group'],
            video=j['video'],success=False,error_type='NotAttempted',failure_stage=stop,duration_seconds=None,new_tokens={})))
    del model,processor
    save(E/'run01/worker.json',dict(plan=bound,records=rows,cases_attempted=attempted,load_seconds=load_seconds,
        stop_reason=stop,labels=sum(read(v)['success'] for v in rows),formal_training_allowed=False))

def run(args):
    r,g=helpers();assert args.gpu_authorized and Path(sys.argv[0]).is_absolute()
    started=time.monotonic()
    p,bound=verify(args.plan,args.sha)
    prior_label=read(p['authorization'])['prior_new_label_GPUh']
    receipt=read(record(E/'publication.remote_bytes.json'))
    assert receipt['remote_equal'] and receipt['public_tests_passed']>0
    import run_teacher_label_semantic_review as review
    review.publication(p,receipt['commit'],str(E/'publication.remote_bytes.json'),time.monotonic()+120)
    initial={u:g.query_modes()[u] for u in p['gpu_uuids']}
    assert sorted(v['index'] for v in initial.values())==([2,3] if BOUNDED_DECODE else list(range(4)))
    assert all(v['mode'] in ('Default','Exclusive_Process') for v in initial.values())
    assert time.monotonic()-started<CAP-120, 'Prelaunch checks must fit full lifecycle cap'
    output=E/'run01';output.mkdir(mode=0o700)
    save(output/'attempt.claim',dict(started_monotonic=started,plan=bound))
    leases=[];child=None;error=None;released=False;restored=False;signals=[]
    def interrupted(*_):raise TimeoutError('Finite label cap or owned termination')
    old={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,max(.01,CAP-60-(time.monotonic()-started)))
    try:
        lease_dir=Path('/dockerdata/wm_gpu_exclusivity');lease_dir.mkdir(mode=0o700,exist_ok=True)
        for u in p['gpu_uuids']:
            f=lease_dir/(u+'.json')
            with f.open('x') as out:json.dump(dict(gpu_uuid=u,expires_epoch=time.time()+CAP,owner_pid=os.getpid()),out)
            f.chmod(0o600);leases.append(f)
        for u in p['gpu_uuids']:
            subprocess.run(['nvidia-smi','-i',u,'-c','3'],check=True,capture_output=True,timeout=10)
            for pid in g.apps().get(u,set()):
                v=g.signal_holder(pid,initial[u]['index']);assert v is not None,'Unknown GPU owner; never signal'
                signals.append(v)
        assert all(not g.apps().get(u) for u in p['gpu_uuids'])
        process=r.helpers()[1];parent=process.identity(os.getpid())
        save(output/'launch.json',dict(parent=parent,plan=bound,started_monotonic=started))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(p['gpu_uuids']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
        with (output/'worker.log').open('xb') as log:
            command=[PYTHON,str(Path(__file__).resolve()),'worker','--plan',args.plan,'--sha',args.sha]
            if WORD_ACTOR_DECODE:command.insert(2,'--word-actor-decode')
            elif BOUNDED_DECODE:command.insert(2,'--bounded-decode')
            elif BOUNDARY_DECODE:command.insert(2,'--boundary-decode')
            elif SCHEMA_DECODE:command.insert(2,'--schema-decode')
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            child.wait(timeout=max(1,CAP-120-(time.monotonic()-started)))
        assert child.returncode==0,'Worker failed; no retry'
        w=read(record(output/'worker.json'));assert w['cases_attempted']==32 and w['labels']==32,'Fixed32 structural gate failed'
    except BaseException as exc:error=type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        queries=[g.apps(),g.apps()];released=all(not q.get(u) for q in queries for u in p['gpu_uuids'])
        for u,v in initial.items():subprocess.run(['nvidia-smi','-i',u,'-c','0' if v['mode']=='Default' else '3'],check=True,capture_output=True,timeout=10)
        restored=all(g.query_modes()[u]['mode']==v['mode'] for u,v in initial.items())
        for f in leases:
            assert json.loads(f.read_text())['owner_pid']==os.getpid();f.unlink()
        for j in p['jobs']:
            f=output/('label'+str(j['index'])+'.json')
            if not f.exists():save(f,dict(index=j['index'],group=j['group'],video=j['video'],success=False,
                error_type='Unrecorded',failure_stage='no_terminal_case_evidence',duration_seconds=None,new_tokens={}))
        labels=[read(record(output/('label'+str(i)+'.json'))) for i in range(32)]
        elapsed=time.monotonic()-started
        result=dict(success=error is None and released and restored and elapsed<=CAP,
            cases_denominator=32,labels=sum(v['success'] for v in labels),errors=sum(not v['success'] for v in labels),
            terminal=True,error_type=error,elapsed_seconds=elapsed,actual_GPUh=CARDS*elapsed/3600,
            charged_upper_GPUh=max(UPPER_GPUH,CARDS*elapsed/3600),prior_new_label_GPUh=prior_label,
            cumulative_new_label_upper_GPUh=prior_label+max(UPPER_GPUH,CARDS*elapsed/3600),released=released,modes_restored=restored,
            holder_signals=signals,semantic_approval=False,formal_training_allowed=False,sota_proven=False)
        save(output/'terminal.json',result)
        for s,h in old.items():signal.signal(s,h)
    assert result['success']

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--schema-decode',action='store_true');parser.add_argument('--boundary-decode',action='store_true');parser.add_argument('--bounded-decode',action='store_true');parser.add_argument('--word-actor-decode',action='store_true');parser.add_argument('command',choices=['prepare','verify','run','worker'])
    parser.add_argument('--output');parser.add_argument('--plan');parser.add_argument('--sha');parser.add_argument('--gpu-authorized',action='store_true');args=parser.parse_args()
    assert sum((args.schema_decode,args.boundary_decode,args.bounded_decode,args.word_actor_decode))<=1
    if args.word_actor_decode:configure_word_actor()
    elif args.bounded_decode:configure_bounded()
    elif args.boundary_decode:configure_boundary()
    elif args.schema_decode:configure_schema()
    if args.command=='prepare':print(json.dumps(prepare(args.output),sort_keys=True))
    elif args.command=='verify':verify(args.plan,args.sha);print('CPU protocol verified; no GPU allocation')
    elif args.command=='worker':worker(args)
    else:run(args)
