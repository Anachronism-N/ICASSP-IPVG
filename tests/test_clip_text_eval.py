import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
spec=importlib.util.spec_from_file_location('clip_text_eval',Path(__file__).resolve().parents[1]/'scripts/clip_text_eval.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unit_cosine_keeps_negative_values_and_no_logit_scale():
    result=module.mean_cosine(np.tile([3.,0.],(8,1)),[-2.,0.])
    assert result=={'clip_frame_text_cosine_mean':-1.,'clip_frame_text_cosine_min':-1.}
    assert module.mean_cosine(np.tile([3.,0.],(8,1)),[0.,2.])['clip_frame_text_cosine_mean']==0


@pytest.mark.parametrize('images,text',[(np.ones((7,2)),[1,0]),(np.zeros((8,2)),[1,0]),
                                      (np.ones((8,2)),[0,0]),(np.ones((8,2)),[1]),
                                      (np.full((8,2),np.nan),[1,0])])
def test_invalid_embeddings_fail(images,text):
    with pytest.raises(ValueError): module.mean_cosine(images,text)
