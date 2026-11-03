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
