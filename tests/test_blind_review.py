import importlib.util
from pathlib import Path
import sys
import pytest
from ipvg.common import write_json,load_json,sha256_file
spec=importlib.util.spec_from_file_location('blindpack',Path(__file__).resolve().parents[1]/'scripts/prepare_blind_review.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_neutral_media_and_no_manufactured_human_scores(tmp_path,monkeypatch):
    image=tmp_path/'reference.png';image.write_bytes(b'image fixture')
    video=tmp_path/'method_named_video.mp4';video.write_bytes(b'video fixture')
    rows=[]
    for i in range(16):
        for group in ('baseline','lr1','lr5','alpha05','alpha025','structure0'):
            rows.append({'sample_id':group+'_'+str(i),'base_sample_id':str(i),'group':group,
                         'video_path':str(video),'video_sha256':sha256_file(video),
                         'reference_image':str(image),'reference_sha256':sha256_file(image),
                         'prompt':'Move <script>alert(1)</script>'})
    manifest=tmp_path/'manifest.json';write_json(manifest,{'samples':rows})
    out=tmp_path/'private_pack'
    monkeypatch.setattr(sys,'argv',['blind','--manifest',str(manifest),'--output-dir',str(out)])
    module.main()
    form=load_json(out/'review_form.json')
    assert form['human_completed'] is False and len(form['reviews'])==80
    assert all(r['human_scores'] is None and r['reviewer_confidence'] is None for r in form['reviews'])
    page=(out/'index.html').read_text()
    assert '<script>alert(1)</script>' not in page and '&lt;script&gt;' in page
    assert 'method_named_video' not in page and 'baseline' not in page
    assert len(load_json(out/'private_method_key.json')['key'])==80
    with pytest.raises(FileExistsError):module.main()
