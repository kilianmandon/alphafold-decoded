import torch
from torch import nn

from kilian.feature_extraction.InputEmbedder import InputEmbedder, ExtraMsaEmbedder
from kilian.feature_extraction.RecyclingEmbedder import RecyclingEmbedder
from kilian.feature_extraction.ExtraMsaStack import ExtraMsaStack
from kilian.feature_extraction.EvoformerStack import EvoformerStack
from kilian.structure_module.structure_module import StructureModule

class Model(nn.Module):
    
    def __init__(self, c_m=256, c_z=128, c_e=64, tf_dim=22, c_s=384):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_e = c_e
        self.c_s = c_s

        self.input_embedder = InputEmbedder(c_m, c_z, tf_dim)
        self.extra_msa_embedder = ExtraMsaEmbedder(25, c_e)
        self.recycling_embedder = RecyclingEmbedder(c_m, c_z)
        self.extra_msa_stack = ExtraMsaStack(c_e, c_z, num_blocks=4)
        self.evoformer = EvoformerStack(c_m, c_z, num_blocks=48)

        self.structure_module = StructureModule(c_s, c_z)

    def forward(self, batch):
        N_cycle = batch['msa_feat'].shape[-1]
        N_seq, N_res = batch['msa_feat'].shape[-4:-2]
        batch_shape = batch['msa_feat'].shape[:-4]
        device = batch['msa_feat'].device

        c_m = self.c_m
        c_z = self.c_z
        c_s = self.c_s

        prev_m = torch.zeros(batch_shape+(N_seq, N_res, c_m), device=device)
        prev_z = torch.zeros(batch_shape+(N_res,N_res,c_z), device=device)
        prev_pseudo_beta_x = torch.zeros((N_res, 3), device=device)

        outputs = {}

        for i in range(N_cycle):
            print(f'Starting iteration {i}...')
            current_batch = {
                key: value[...,i] for key, value in batch.items()
            }
            m, z = self.input_embedder(current_batch)
            m, z = self.recycling_embedder(m, z, prev_m, prev_z, prev_pseudo_beta_x)

            e = self.extra_msa_embedder(current_batch)
            z = self.extra_msa_stack(e, z)
            del e

            m, z, s = self.evoformer(m, z)
            torch.save(m, 'kilian/test_outputs/m_evo.pt')
            torch.save(z, 'kilian/test_outputs/z_evo.pt')
            torch.save(s, 'kilian/test_outputs/s_evo.pt')

            F = current_batch['target_feat'].argmax(dim=-1) - 1
            structure_output = self.structure_module(s, z, F)
            prev_m = m
            prev_z = z
            prev_pseudo_beta_x = structure_output['pseudo_beta_positions']

            for key,value in structure_output.items():
                if key in outputs:
                    outputs[key].append(value)
                else:
                    outputs[key] = [value]

        outputs = {
            key: torch.stack(value, dim=0) for key, value in outputs.items()
        }
        return outputs