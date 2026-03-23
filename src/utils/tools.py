import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from VadCLIP.src.utils.tools import (
    process_feat,
    process_split,
    pad,
    uniform_extract,
    random_extract,
    get_batch_mask,
    get_batch_label,
    get_prompt_text,
)
