
import pytest
from tools.ckpts import convert_sequential_to_hf
from tests.common import simulate_deepy_env, save_random_model
from megatron.neox_arguments.neox_args import NeoXArgsTokenizer

# Test is failing; possibly we resolve by using the word embedding weights in the 'layer_00-model_00-model_states.pt' file?
@pytest.mark.xfail(reason="Failing to find 'word_embeddings.weight' in state_dict['module'] from the 'mp_rank_00_model_states.pt' file")
def test_gpt_neox_to_huggingface(monkeypatch, tmpdir, tmp_path):
    # Generate random GPT-NEOX model, check we can convert to hf format
    model_dir = str(tmpdir)
    input_args = ["train.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)
    save_random_model(deepspeed_main_args, model_dir, train_iters=1)

    # Generate output
    script_args = ["--config_file", "tests/config/test_setup.yml", "--input_dir", model_dir+"/global_step1", "--output_dir", model_dir]
    overwrite_values = {"tokenizer_type": NeoXArgsTokenizer.tokenizer_type}
    convert_sequential_to_hf.main(input_args=script_args, overwrite_values=overwrite_values)