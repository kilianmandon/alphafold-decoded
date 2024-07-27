import torch
from torch.distributions import Categorical
#
#    return { 'msa_aatype': unique_seqs, 'msa_deletion_count': deletion_count_matrix, 'aa_distribution': aa_distribution}
#     Returns:
#        A dictionary containing:
#            * msa_aatype: A PyTorch tensor of one-hot encoded amino acid sequences
#                  of shape (N_seq, N_res, 22), where N_seq is the number of unique 
#                  sequences (with deletions removed) and N_res is the length of the sequences. 
#                  The dimension 22 corresponds to the 20 amino acids, an unknown amino acid 
#                  token, and a gap token. 
#            * msa_deletion_count: A tensor of shape (N_seq, N_res) where 
#                  each element represents the number of deletions occurring before 
#                  the corresponding residue in the MSA.
#            * aa_distribution: A tensor of shape (N_res, 22) containing the 
#                  overall amino acid distribution at each residue position 
#                  across the MSA. 

msa_aatype = None
msa_deletion_count = None
aa_distribution = None
seqs_without_deletions = None
seqs = None
one_hot_encode_aatype = lambda x: None
count_lowercase_chars = lambda x: None
randperm = None
N = None
feature = None
max_msa_clusters = None

# Shape (N_seq, N_res, 22), one-hot-encoded
msa_aatype = one_hot_encode_aatype(seqs_without_deletions)

# Shape (N_seq, N_res)
msa_deletion_count = count_lowercase_chars(seqs)

# Shape (N_res, 22)
aa_distribution = torch.mean(msa_aatype, dim=0)

# e.g. [0, 3, 5, 4, 2, 1]
shuffled_inds = [0] + randperm(1, N)

# e.g. [0, 3]
cluster_inds = shuffled_inds[:max_msa_clusters]

# e.g. [5, 4, 2, 1]
extra_inds = shuffled_inds[max_msa_clusters:]

feature = feature[cluster_inds]
extra_feature = feature[extra_inds]

concatenate = None
features = None
N_clust, N_res = None, None

# Shape (22), needs broadcasting
uniform = [1/20]*20 + [0,0] 

# Shape (N_clust, N_res, 22)
from_profile = features['profile']

# Shape (N_clust, N_res, 22)
no_replacement = features['msa_aatype']

# Shape (N_clust, N_res, 22)
categories = 0.1 * uniform + 0.1 * from_profile + \
    0.1 * no_replacement

# Shape (N_clust, N_res, 23), flatten to (N_clust*N_res, 23)
categories = concatenate(categories, [0.7])
replacement = Categorical(categories)

where_to_replace = torch.rand((N_clust, N_res)) < 0.15


agreement = None

# agreement has shape (N_clust, N_extra)
# assignment has shape (N_extra)
assignment = torch.argmax(agreement,dim=0)
features['cluster_assignment'] = assignment

# assignment_counts has shape (N_clust)
assignment_counts = torch.bincount(assignment, minlength=N_clust)
features['cluster_assignment_counts'] = assignment_counts

cluster_assignment_count = None
cluster_assignment = None

cluster_sum = torch.scatter_add(feature, dim=0, index=cluster_assignment, \
                                src=extra_feature)

cluster_average = cluster_sum / (cluster_assignment_count + 1)