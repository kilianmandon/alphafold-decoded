import math
import torch

N = 3
c_m = 4
c_z = 5
c = 6
N_head = 7
N_seq = 8
N_extra = 9
N_res = 35
msa_feat_dim = 10
tf_dim = 11
f_e = 12
c_e = 13

feature_shapes = {
    'm': (N_seq, N_res, c_m),
    'z': (N_res, N_res, c_z),
    'msa_feat': (N_seq, N_res, msa_feat_dim),
    'target_feat': (N_res, tf_dim),
    'residue_index': (N_res,),
    'extra_msa_feat': (N_extra, N_res, f_e),
    'e': (N_extra, N_res, c_e),
    'x': (N_res, 3),
}

batched_feature_shapes = {
    key: (N,) + value
    for key, value in feature_shapes.items()
}

test_inputs = {
    key: torch.linspace(-2, 2, math.prod(shape)).reshape(shape)
    for key, shape in feature_shapes.items()
}

test_inputs['residue_index'] = torch.arange(N_res)

test_inputs['batch'] = {
        'msa_feat': test_inputs['msa_feat'],
        'target_feat': test_inputs['target_feat'],
        'residue_index': test_inputs['residue_index'],
        'extra_msa_feat': test_inputs['extra_msa_feat'],
}



def test_module_shape(module, test_name, control_folder, overwrite_results=False):
    param_shapes = {name: param.shape for name, param in module.named_parameters()}

    shapes_path = f'{control_folder}/{test_name}_param_shapes.pt'

    if overwrite_results:
        torch.save(param_shapes, shapes_path)

    expected_shapes = torch.load(shapes_path)

    keys_in_dict1_only = set(param_shapes.keys()) - set(expected_shapes.keys())
    keys_in_dict2_only = set(expected_shapes.keys()) - set(param_shapes.keys())

    if keys_in_dict1_only or keys_in_dict2_only:
        error_message = "Parameter key set mismatch:\n"
        if keys_in_dict1_only:
            error_message += f"Unexpected keys: {keys_in_dict1_only}\n"
        if keys_in_dict2_only:
            error_message += f"Missing keys: {keys_in_dict2_only}\n"
        raise AssertionError(error_message)

    for name, param_shape in param_shapes.items():
        assert param_shape == expected_shapes[name], f'Invalid shape for parameter {name}.'
            

    
def controlled_execution(module, inp, method):
    was_training = module.training
    module.eval()
    with torch.no_grad():
        original_params = [param.clone() for param in module.parameters()]
        for param in module.parameters():
            param.copy_(torch.linspace(-1, 1, param.numel()).reshape(param.shape))

        out = method(*inp)

        for orig_param, param in zip(original_params, module.parameters()):
            param.copy_(orig_param)

    if was_training:
        module.train()

    return out

def controlled_forward(module, inp):
    return controlled_execution(module, inp, lambda *x: module(*x))
    

def test_module_method(module, test_name, input_names, output_names, control_folder, method, overwrite_results=False):
    if not isinstance(input_names, (tuple, list)):
        input_names = (input_names,)

    if not isinstance(output_names, (tuple, list)):
        output_names = (output_names,)

    non_batched_inps = tuple(test_inputs[name] for name in input_names)
    batched_inps = []
    for name in input_names:
        inp = test_inputs[name]
        if isinstance(inp, dict):
            batched_inp = {
                key: value.broadcast_to(batched_feature_shapes[key])
                for key, value in inp.items()
            }
        else:
            batched_inp = inp.broadcast_to(batched_feature_shapes[name])
        batched_inps.append(batched_inp)

    batched_inp = tuple(batched_inp)

    with torch.no_grad():
        non_batched_out = controlled_execution(module, non_batched_inps, method)
        batched_out = controlled_execution(module, batched_inps, method)

    if not isinstance(non_batched_out, (tuple, list)):
        non_batched_out = (non_batched_out,)
        batched_out = (batched_out,)

    out_file_names = [f'{control_folder}/{test_name}_{out_name}.pt' for out_name in output_names]

    if overwrite_results:
        for out, out_name in zip(non_batched_out, out_file_names):
            torch.save(out, out_name)

    for out, out_file_name, out_name in zip(non_batched_out, out_file_names, output_names):
        expected_out = torch.load(out_file_name)
        assert torch.allclose(out, expected_out, atol=1e-5), f'Problem with output {out_name} in test {test_name} in non-batched check.'

    for out, out_file_name, out_name in zip(batched_out, out_file_names, output_names):
        expected_out = torch.load(out_file_name)
        expected_out_batch_shape = (N,) + expected_out.shape
        expected_out = expected_out.unsqueeze(0).broadcast_to(expected_out_batch_shape)
        assert torch.allclose(out, expected_out, atol=1e-5), f'Problem with output {out_name} in test {test_name} in batched check.'

    
def test_module_forward(module, test_name, input_names, output_names, control_folder, overwrite_results=False):
    test_module_method(module, test_name, input_names, output_names, control_folder, lambda *x: module(*x), overwrite_results=overwrite_results)


