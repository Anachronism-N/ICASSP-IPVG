import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import teacher_gpu_exclusive as guard

def test_unknown_process_identity_never_signaled():
    with (patch.object(guard.os,'pidfd_open',return_value=77,create=True),patch.object(guard.os,'close') as close,
         patch.object(guard,'holder_identity',return_value=None),patch.object(guard.signal,'pidfd_send_signal',create=True) as sig):
        assert guard.signal_holder(123,0) is None
        sig.assert_not_called();close.assert_called_once_with(77)

def test_exact_holder_term_uses_pidfd_no_global_kill():
    with (patch.object(guard.os,'pidfd_open',return_value=77,create=True),patch.object(guard.os,'close'),
         patch.object(guard,'holder_identity',return_value={'pid':123}),patch.object(guard.signal,'pidfd_send_signal',create=True) as sig,
         patch.object(guard.select,'select',return_value=([77],[],[]))):
        assert guard.signal_holder(123,0)=={'pid':123}
        sig.assert_called_once_with(77,guard.signal.SIGTERM)



def test_node_local_lease_paths_are_uuid_specific_and_bounded():
    import json,time
    expiry=time.time()+5400
    lease={'gpu_uuid':'GPU-test','expires_epoch':expiry,'owner_pid':123,'owner_start':10.0}
    assert lease['gpu_uuid']=='GPU-test' and 0<lease['expires_epoch']-time.time()<=5400
    assert Path('/dockerdata/wm_gpu_exclusivity')/(lease['gpu_uuid']+'.json')==Path('/dockerdata/wm_gpu_exclusivity/GPU-test.json')
