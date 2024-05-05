import torch
import modelcif
import modelcif.model
import modelcif.dumper
import io
from geometry.residue_constants import atom_types

def parameter_renaming(weights):
    name_map = {
        'core.msa_transition': 'msa_transition',
        'core.outer_product_mean': 'outer_product_mean',
        'msa_att_col._msa_att': 'msa_att_col',
        'transition.layers.0.linear_1': 'transition.linear_1',
        'transition.layers.0.linear_2': 'transition.linear_2',
        'transition.layers.0.linear_3': 'transition.linear_3',
    }

    for key in list(weights.keys()):
        new_key = None
        for sub in name_map.keys():
            if sub in key:
                new_key = key.replace(sub, name_map[sub])

        if new_key is not None:
            weights[new_key] = weights.pop(key)

    return weights

def ipa_kv_split(weights):
    for key in list(weights.keys()):
        if 'structure_module.ipa.linear_kv.' in key:
            value = weights[key]
            N_head, c = 12, 16
            value = value.view((N_head, 2*c) + value.shape[1:])
            k, v = value.split(c, dim=1)
            if 'weight' in key:
                weights['structure_module.ipa.linear_k.weight'] = k.flatten(end_dim=1)
                weights['structure_module.ipa.linear_v.weight'] = v.flatten(end_dim=1)
            else:
                weights['structure_module.ipa.linear_k.bias'] = k.flatten(end_dim=1)
                weights['structure_module.ipa.linear_v.bias'] = v.flatten(end_dim=1)

            weights.pop(key)

    return weights

def ipa_kv_points_split(weights):
    for key in list(weights.keys()):
        if 'structure_module.ipa.linear_kv_points.' in key:
            value = weights[key]
            N_head, query_points, value_points = 12, 4, 8
            value = value.view((3*N_head, query_points+value_points) + value.shape[1:])
            k, v = value.split([query_points, value_points], dim=1)
            if 'weight' in key:
                weights['structure_module.ipa.linear_k_points.weight'] = k.flatten(end_dim=1)
                weights['structure_module.ipa.linear_v_points.weight'] = v.flatten(end_dim=1)
            else:
                weights['structure_module.ipa.linear_k_points.bias'] = k.flatten(end_dim=1)
                weights['structure_module.ipa.linear_v_points.bias'] = v.flatten(end_dim=1)
            weights.pop(key)

    return weights

def angle_cos_sin_flip(weights):
    for key in list(weights.keys()):
        if 'structure_module.angle_resnet.linear_out' in key:
            value = weights[key]
            swap = value[::2].clone()
            value[::2] = value[1::2]
            value[1::2] = swap
            weights[key] = value

    return weights

def target_feat_drop_pad(weights):
    for key in list(weights.keys()):
        if 'linear_tf' in key and 'weight' in key:
            weights[key] = weights[key][:, 1:]

    return weights

def remove_unused(weights):
    for key in list(weights.keys()):
        if 'template' in key or 'aux' in key:
            weights.pop(key)
    return weights
            
            
def load_openfold_weights(path):
    openfold_weights = torch.load(path, map_location='cpu')

    openfold_weights = parameter_renaming(openfold_weights)
    openfold_weights = ipa_kv_split(openfold_weights)
    openfold_weights = ipa_kv_points_split(openfold_weights)
    openfold_weights = angle_cos_sin_flip(openfold_weights)
    openfold_weights = target_feat_drop_pad(openfold_weights)
    openfold_weights = remove_unused(openfold_weights)

    return openfold_weights

    
def to_modelcif(atom_positions, atom_mask, sequence):
    atom_positions = atom_positions.to('cpu').numpy()
    atom_mask = atom_mask.to('cpu').numpy()
    n = atom_positions.shape[0]
    system = modelcif.System(title='AlphaFold prediction')
    entity = modelcif.Entity(sequence, description='Model subunit')
    asym_unit = modelcif.AsymUnit(entity, details='Model subunit A', id='A')
    modeled_assembly = modelcif.Assembly([asym_unit], name='Modeled assembly')

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            for i in range(n):
                for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
                    if not mask:
                        continue
                    element = atom_name[0]
                    yield modelcif.model.Atom(
                        asym_unit=asym_unit,
                        type_symbol=element,
                        seq_id=i+1,
                        atom_id=atom_name,
                        x=pos[0], y=pos[1], z=pos[2],
                        het=False,
                        occupancy=1.00
                    )

    model = _MyModel(assembly=modeled_assembly, name='Model')
    model_group = modelcif.model.ModelGroup([model], name='All models')
    system.model_groups.append(model_group)
    fh = io.StringIO()
    modelcif.dumper.write(fh, [system])

    return fh.getvalue()