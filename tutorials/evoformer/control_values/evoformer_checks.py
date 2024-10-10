import math
import torch

N = 3
c_m = 4
c_z = 5
c = 6
N_head = 7
N_seq = 8
N_res = 9

test_m_shape = (N_seq, N_res, c_m)
test_z_shape = (N_res, N_res, c_z)
test_m_shape_batched = (N,) + test_m_shape
test_z_shape_batched = (N,) + test_z_shape

test_m = torch.linspace(-2, 2, math.prod(test_m_shape)).reshape(test_m_shape)
test_z = torch.linspace(-2, 2, math.prod(test_z_shape)).reshape(test_z_shape)
test_m_batch = torch.linspace(-2, 2, math.prod(test_m_shape_batched)
                              ).reshape(test_m_shape_batched)
test_z_batch = torch.linspace(-2, 2, math.prod(test_z_shape_batched)
                              ).reshape(test_z_shape_batched)

test_inputs = {
    'm': (test_m.double(), test_m_batch.double()),
    'z': (test_z.double(), test_z_batch.double()),
}

def test_module_shape(module, test_name, control_folder):
    param_shapes = {name: param.shape for name, param in module.named_parameters()}

    shapes_path = f'{control_folder}/{test_name}_param_shapes.pt'
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
            

    
def controlled_forward(module, inp):
    was_training = module.training
    module.eval()
    module.double()
    with torch.no_grad():
        original_params = [param.clone() for param in module.parameters()]
        for param in module.parameters():
            param.copy_(torch.linspace(-1, 1, param.numel()).reshape(param.shape))

        out = module(*inp)

        for orig_param, param in zip(original_params, module.parameters()):
            param.copy_(orig_param)

    if was_training:
        module.train()

    return out


    


def test_module(module, test_name, input_names, output_names, control_folder):
    if not isinstance(input_names, (tuple, list)):
        input_names = (input_names,)

    if not isinstance(output_names, (tuple, list)):
        output_names = (output_names,)

    non_batched_inps = (test_inputs[name][0] for name in input_names)
    batched_inps = (test_inputs[name][1] for name in input_names)

    with torch.no_grad():
        non_batched_out = controlled_forward(module, non_batched_inps)
        batched_out = controlled_forward(module, batched_inps)

    if not isinstance(non_batched_out, (tuple, list)):
        non_batched_out = (non_batched_out,)
        batched_out = (batched_out,)

    out_file_names = [f'{control_folder}/{test_name}_{out_name}.pt' for out_name in output_names]
    out_file_names_batched = [f'{control_folder}/{test_name}_{out_name}_batched.pt' for out_name in output_names]
    for out, out_file_name, out_name in zip(non_batched_out, out_file_names, output_names):
        expected_out = torch.load(out_file_name)
        max_abs_err = (out.double()-expected_out.double()).abs().max()
        max_rel_err = ((out.double()-expected_out.double()).abs() / (torch.maximum(out.double().abs(), expected_out.double().abs())+1e-8)).max()
        assert torch.allclose(out, expected_out, atol=1e-5), f'Problem with output {out_name} in test {test_name} in batched check. Maximum absolute error: {max_abs_err}. Maximum relative error: {max_rel_err}.'

    for out, out_file_name, out_name in zip(batched_out, out_file_names_batched, output_names):
        expected_out = torch.load(out_file_name)
        max_abs_err = (out.double()-expected_out.double()).abs().max()
        max_rel_err = ((out.double()-expected_out.double()).abs() / (torch.maximum(out.double().abs(), expected_out.double().abs())+1e-8)).max()
        assert torch.allclose(out, expected_out, atol=1e-5), f'Problem with output {out_name} in test {test_name} in batched check. Maximum absolute error: {max_abs_err}. Maximum relative error: {max_rel_err}.'

    



