import torch
import re
from torch import nn
from structure_module.residue_constants import restype_order_with_x, restype_order_with_x_and_gap

def create_features_from_a3m(file_name):
    """
    Creates feature representations for an MSA from its A3M file.

    This function orchestrates a sequence of transformations on the raw MSA sequences to 
    produce features suitable for protein structure prediction.

    Args:
        file_name: Path to the A3M file containing the MSA sequences.

    Returns:
        A dictionary containing the following feature representations for the MSA:
           * msa_feat: A tensor containing the final MSA feature representation.
           * extra_msa_feat: A tensor containing the final extra MSA feature representation.
           * target_feat: A tensor containing a one-hot encoded representation of the 
                          target protein sequence (excluding gaps and masked tokens).
           * residue_index: A tensor containing the residue indices (0, 1, ..., N_res-1). 
    """

    msa_feat = None
    extra_msa_feat = None
    target_feat = None
    residue_index = None

    ######################################################################
    # TODO:
    # 1. **Load A3M File:**
    #     * Use `load_a3m_file` to read the A3M file and extract a list of raw MSA sequences.
    # 2. **Initial Features:**
    #     * Call `initial_data_from_seqs` to create initial feature representations from the raw  sequences. This will include one-hot encoded amino acids,  deletion counts, and an amino acid distribution ('aa_distribution').
    # 3. **Feature Transformations:**  
    #     * Define a list of transformation functions.  This should include `select_cluster_centers`, `mask_cluster_centers`, `cluster_assignment`, `summarize_clusters`, and `crop_extra_msa`.  **Importantly, ensure the random seeds are set as indicated in your code.**
    #     * Iterate through the `transforms` list, applying each transformation function in  sequence to the `features` dictionary.  
    # 4. **Final Features:**
    #      * Calculate the final MSA feature representations using `calculate_msa_feat`.
    #      * Calculate the final extra MSA feature representations using `calculate_extra_msa_feat`.
    # 5. **Target Features and Indices:**
    #     * Employ `onehot_encode_aa_type` on the first sequence in the MSA (`seqs[0]`), excluding gaps, to create the `target_feat`.
    #     * Create a `residue_index` tensor using `torch.arange(len(seqs[0]))`.
    ######################################################################

    seqs = load_a3m_file(file_name)
    features = initial_data_from_seqs(seqs)

    transforms = [
        lambda x: select_cluster_centers(x, seed=0),
        lambda x: mask_cluster_centers(x, seed=1),
        cluster_assignment,
        summarize_clusters,
        lambda x: crop_extra_msa(x, seed=2)
    ]

    for transform in transforms:
        features = transform(features)

    msa_feat = calculate_msa_feat(features)
    extra_msa_feat = calculate_extra_msa_feat(features)

    # Note: In the official openfold implementation, the one-hot encoding
    # is padded with zeroes for compatibility with domain datasets for the target_feat
    target_feat = onehot_encode_aa_type(seqs[0], include_gap_token=False)
    residue_index = torch.arange(len(seqs[0]))

    # End of your code

    return {
        'msa_feat': msa_feat,
        'extra_msa_feat': extra_msa_feat,
        'target_feat': target_feat,
        'residue_index': residue_index
    }


def onehot_encode_aa_type(seq, include_gap_token=False):
    """
    Converts a protein sequence into one-hot encoding. X represents an unkown amino acid.

    Args:
        seq:  A string representing the amino acid sequence using single-letter codes.
        include_gap_token: If True, includes an extra token ('-') in the encoding to 
                           represent gaps.

    Returns: 
        A PyTorch tensor of shape (N_res, 22) if `include_gap_token` is True, 
        or shape (N_res, 21) otherwise.  Here, N_res is the length of the sequence.
    """
    restype_order = restype_order_with_x if not include_gap_token else restype_order_with_x_and_gap
    encoding = None

    ######################################################################
    # TODO:
    # 1. Obtain the correct numerical index for each amino acid in the input 
    #    sequence 'seq' using the  'restype_order' variable.
    # 2. Apply PyTorch's `nn.functional.one_hot` encoding the numerical indices
    #    to create the final one-hot representation. 
    ######################################################################

    
    sequence_inds = torch.tensor([restype_order[a] for a in seq])
    encoding = nn.functional.one_hot(sequence_inds, num_classes=len(restype_order))

    ######################################################################
    # END OF YOUR CODE
    ######################################################################

    return encoding


def load_a3m_file(file_name: str):
    """
    Loads an A3M (multiple sequence alignment) file and extracts the raw amino acid sequences.

    Args:
        file_name: Path to the A3M file.

    Returns:
        A list of strings where each string represents an individual protein sequence from the input MSA.
    """
    ######################################################################
    # TODO: 
    # 1. Read the A3M file line by line.
    # 2. Identify lines that start with '>' as sequence description lines.
    # 3. Extract the sequence from the lines following each description line.
    # 4. Strip them to remove leading and trailing whitespace
    ###################################################################### 


    with open(file_name, 'r') as f:
        lines = f.readlines()

    description_line_indices = [i for i,l in enumerate(lines) if l.startswith('>')]
    seqs = [lines[i+1].strip() for i in description_line_indices]
    return seqs

def initial_data_from_seqs(seqs: list[str]):
    """
    Processes raw sequences from an A3M file to extract initial feature representations.

    Args:
        seqs: A list of amino acid sequences loaded from the A3M file. 
              Sequences are represented with single-letter amino acid codes.
              Lowercase letters represent deletions.

    Returns:
        A dictionary containing:
            * unique_seqs: A PyTorch tensor of one-hot encoded amino acid sequences
                  of shape (N_seq, N_res, 22), where N_seq is the number of unique 
                  sequences (with deletions removed) and N_res is the length of the sequences. 
                  The dimension 22 corresponds to the 20 amino acids, an unknown amino acid 
                  token, and a gap token. 
            * deletion_count_matrix: A tensor of shape (N_seq, N_res) where 
                  each element represents the number of deletions occurring before 
                  the corresponding residue in the MSA.
            * aa_distribution: A tensor of shape (N_res, 22) containing the 
                  overall amino acid distribution at each residue position 
                  across the MSA.  
    """

    unique_seqs = None
    deletion_count_matrix = None
    aa_dist = None

    ######################################################################
    # TODO: 
    # 1. Calculate the 'deletion_count_matrix':
    #    * Initialize an empty list of lists to store deletion counts.
    #    * Iterate through the sequences in 'seqs':
    #       * Create a list to track deletions for the current sequence.
    #       * Iterate through letters, counting lowercase letters as deletions.
    #       * Append the deletion count list to the main 'deletion_count_matrix' only 
    #         if the sequence (after removing deletions) has not been seen before. 
    #    * Convert 'deletion_count_matrix' into a PyTorch tensor. 
    # 2. Identify 'unique_seqs':  
    #    * Create an empty list to store unique sequences.
    #    * Iterate through the sequences in 'seqs':
    #       * Remove lowercase letters (deletions) from the sequence.
    #       * If the sequence (without deletions) is not already in the 'unique_seqs'
    #         list, add it.
    #    * Apply the `onehot_encode_aa_type` function to each sequence in 'unique_seqs' 
    #      to get a tensor of shape (N_seq, N_res, 22) representing the one-hot encoded amino acids. 
    # 3. Compute 'aa_dist':
    #    * Average the one-hot encoded 'unique_seqs' tensor across the first dimension
    #      (representing sequences) to calculate the amino acid distribution.
    ######################################################################

    deletion_count_matrix = []
    unique_seqs = []
    for seq in seqs:
        deletion_count_list = []
        deletion_counter = 0
        for letter in seq:
            if letter.islower():
                deletion_counter += 1
            else:
                deletion_count_list.append(deletion_counter)
                deletion_counter=0
        seq_without_deletion = re.sub('[a-z]', '', seq)

        if seq_without_deletion in unique_seqs:
            continue

        unique_seqs.append(seq_without_deletion)
        deletion_count_matrix.append(deletion_count_list)
    
    unique_seqs = torch.stack([onehot_encode_aa_type(seq, include_gap_token=True) for seq in unique_seqs], dim=0)
    unique_seqs = unique_seqs.float()
    deletion_count_matrix = torch.tensor(deletion_count_matrix).float()
    aa_dist = unique_seqs.float().mean(dim=0)

    # End of your code

    return { 'msa_aatype': unique_seqs, 'msa_deletion_count': deletion_count_matrix, 'aa_distribution': aa_dist}

def select_cluster_centers(features, max_msa_clusters=512, seed=None):
    """
    Selects representative sequences as cluster centers from the MSA to  
    reduce redundancy.

    Args:
        features: A dictionary containing feature representations of the MSA.
        max_msa_clusters: The maximum number of cluster centers to select.
        seed: An optional integer seed for the random number generator. 
              Use this to ensure reproducibility.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' and 'msa_deletion_count' features to contain 
              data for the cluster centers only.  
            * Adding 'extra_msa_aatype' and 'extra_msa_deletion_count' features
              to hold the data for the remaining (non-center) sequences. 
    """

    N_seq, N_res = features['msa_aatype'].shape[:2]
    MSA_FEATURE_NAMES = ['msa_aatype', 'msa_deletion_count']
    max_msa_clusters = min(max_msa_clusters, N_seq)

    gen = torch.Generator(device=features['msa_aatype'].device)
    if seed is not None:
        gen.manual_seed(seed)

    ######################################################################
    # TODO:
    # 1. **Implement Shuffling:**
    #      * Use  `torch.randperm(N_seq - 1)` with the provided  `gen` (random number generator) 
    #        to shuffle the indices from 1 to (N_seq - 1). Ensure reproducibility if the seed is not None.
    #      * Prepend a 0 to the shuffled indices to include the first sequence.
    # 2. **Split Features:**
    #      * Using the shuffled indices,  split the MSA feature representations (`msa_aatype` and
    #        `msa_deletion_count`) into two sets:
    #          *  The first `max_msa_clusters` sequences will be the cluster centers.
    #          *  The remaining sequences will be stored with keys prefixed by  'extra_'. 
    ######################################################################


    shuffled = torch.randperm(N_seq-1, generator=gen) + 1
    shuffled = torch.cat((torch.tensor([0]), shuffled), dim=0)

    for key in MSA_FEATURE_NAMES:
        extra_key = f'extra_{key}'
        value = features[key]
        features[extra_key] = value[shuffled[max_msa_clusters:]]
        features[key] = value[shuffled[:max_msa_clusters]]

    # End of your code

    return features

def mask_cluster_centers(features, mask_probability=0.15, seed=None):
    """
    Introduces random masking in the cluster center sequences for data augmentation.

    This function modifies the 'msa_aatype' feature within the 'features' dictionary to improve 
    model robustness in the presence of noisy or missing input data.  Masking is inspired by 
    the AlphaFold architecture.

    Args:
        features: A dictionary containing feature representations of the MSA. It is assumed
                  that cluster centers have already been selected.
        mask_probability: The probability of masking out an individual amino acid 
                          in a cluster center sequence.
        seed: An optional integer seed for the random number generator. 
              Use this to ensure reproducibility.

    Modifies:
        The 'features' dictionary in-place by:
            * Updating the 'msa_aatype' feature with masked-out tokens as well as possible 
              replacements based on defined probabilities. 
            * Creating a copy of the original 'msa_aatype' feature with the key 'true_msa_aatype'. 
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_aa_categories = 23 # 20 Amino Acids, Unknown AA, Gap, masked_msa_token
    odds = {
        'uniform_replacement': 0.1,
        'replacement_from_distribution': 0.1,
        'no_replacement': 0.1,
        'masked_out': 0.7,
    }
    gen = torch.Generator(features['msa_aatype'].device)
    if seed is not None:
        gen.manual_seed(seed)
        torch.manual_seed(seed)

    ######################################################################
    # TODO:
    # 1. **Select Modification Candidates:**
    #      * Generate a random mask (tensor of shape (N_clust, N_res) ) where each element is a 
    #        random number between 0 and 1. 
    #      * Select elements where the random number is less than the `mask_probability` for potential modification.
    # 2. **Replacement Logic:**
    #      * Create tensors to represent substitution probabilities:
    #          * `uniform_replacement`: Shape (22,) 
    #             - Set the first 20 elements (amino acids) to `1/20 * odds['uniform_replacement']`.
    #             - Set the last 2 elements (unknown AA and gap) to 0.
    #          * `replacement_from_distribution`: Shape (N_res, 22), calculated from 'features['aa_distribution]'. Scale by `odds['replacement_from_distribution']`
    #          *  `no_replacement`: Shape (N_clust, N_res, 22), use the existing 'features['msa_aatype']' tensor and scale by `odds['no_replacement']`.
    #          * `masked_out`: Shape (N_clust, N_res, 1), all elements are `odds['masked_out']`.
    #      * **Sum** the first three tensors, then **concatenate** with `masked_out` along the last dimension.  This creates 'categories_with_mask_token' of shape (N_clust, N_res, 23)
    #      * Flatten the first two dimensions of 'categories_with_mask_token' for sampling.
    #      * Use  `torch.distributions.Categorical` and the flattened 'categories_with_mask_token' tensor to 
    #        probabilistically determine replacements for the selected residues. 
    #      * Reshape the sampled replacements back to (N_clust, N_res).
    # 3. **Preserve Original Data:**
    #      * Create a copy of the original 'msa_aatype' data under the key 'true_msa_atype'.
    # 4. **Apply Masking:**
    #      * Update the 'msa_aatype' tensor, but *only* for the elements selected in step 1 for modification, with the sampled replacements.  Leave other elements unchanged. 
    ######################################################################


    # uniform_replacement has shape (22,)
    uniform_replacement = torch.tensor([1/20]*20+[0,0]) * odds['uniform_replacement']
    # replacement_from_distribution has shape (N_res, 22)
    replacement_from_distribution = features['aa_distribution'] * odds['replacement_from_distribution']
    # no_replacement has shape (N_clust, N_res, 22)
    no_replacement = features['msa_aatype'] * odds['no_replacement']
    # masked_out has shape (N_clust, N_res, 1)
    masked_out = torch.ones((N_clust, N_res, 1)) * odds['masked_out']

    uniform_replacement = uniform_replacement[None, None, ...].broadcast_to(no_replacement.shape)
    replacement_from_distribution = replacement_from_distribution[None, ...].broadcast_to(no_replacement.shape)

    categories_without_mask_token = uniform_replacement + replacement_from_distribution + no_replacement
    categories_with_mask_token = torch.cat((categories_without_mask_token, masked_out), dim=-1)
    categories_with_mask_token = categories_with_mask_token.reshape(-1, N_aa_categories)
    
    replace_with = torch.distributions.Categorical(categories_with_mask_token).sample()
    replace_with = nn.functional.one_hot(replace_with, num_classes=N_aa_categories)
    replace_with = replace_with.reshape(N_clust, N_res, N_aa_categories)
    replace_with = replace_with.float()

    replace_mask = torch.rand((N_clust, N_res), generator=gen) < mask_probability

    features['true_msa_aatype'] = features['msa_aatype'].clone()
    aatype_padding = torch.zeros((N_clust, N_res, 1))
    features['msa_aatype'] = torch.cat((features['msa_aatype'], aatype_padding), dim=-1)
    features['msa_aatype'][replace_mask] = replace_with[replace_mask]

    # End of your code

    return features

def cluster_assignment(features):
    """
    Assigns sequences in the extra MSA to their closest cluster centers based on Hamming distance.

    Args:
        features: A dictionary containing feature representations of the MSA. 
                  It is assumed that cluster centers have already been selected.

    Returns:
        The updated 'features' dictionary with the following additions:
            * cluster_assignment:  A tensor of shape (N_extra,) containing the indices 
                                  of the assigned cluster centers for each extra sequence.
            * cluster_assignment_counts: A tensor of shape (N_clust,)  where each element indicates 
                                        the number of extra sequences assigned to a cluster center 
                                        (excluding the cluster center itself).
    """
    
    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    ######################################################################
    # TODO:
    # 1. **Prepare Features:**
    #     * Obtain slices of the 'msa_aatype' (shape: N_clust, N_res, 23) and 'extra_msa_aatype' (shape: N_extra, N_res, 22) tensors 
    #       that exclude the  'gap' and 'masked' tokens.  This focuses the calculation on the standard amino acids.
    # 2. **Calculate Agreement:**
    #     * Employ broadcasting and tensor operations on the prepared features to efficiently calculate the number of positions where 
    #       the amino acids in each extra sequence agree with those in each cluster center.  The result will be an 'agreement' tensor 
    #       of shape (N_clust, N_extra).  `torch.einsum` can be a useful tool here. 
    # 3. **Assign Clusters:**
    #     * Use `torch.argmax(agreement, dim=0)` to find the cluster center index with the highest agreement (lowest Hamming distance) for each extra sequence. 
    # 4. **Compute Assignment Counts:** 
    #     * Use `torch.bincount` to efficiently calculate the number of extra sequences assigned to each cluster center (excluding 
    #       the cluster center itself).  Ensure you set the `minlength` parameter appropriately.
    ######################################################################


    msa_aatype = features['msa_aatype'][...,:21]
    extra_msa_aatype = features['extra_msa_aatype'][...,:21]
    agreement = torch.einsum('cra,era->ce', msa_aatype, extra_msa_aatype)
    assignment = torch.argmax(agreement,dim=0)
    features['cluster_assignment'] = assignment

    assignment_counts = torch.bincount(assignment, minlength=N_clust)
    features['cluster_assignment_counts'] = assignment_counts

    # End of your code

    return features

def cluster_average(feature, extra_feature, cluster_assignment, cluster_assignment_count):
    """
    Calculates the average representation of each cluster center by aggregating features 
    from the assigned extra sequences.

    Args:
        feature: A tensor containing feature representations for the cluster centers.
                 Shape: (N_clust, N_res, *)
        extra_feature: A tensor containing feature representations for extra sequences.
                       Shape: (N_extra, N_res, *).  The trailing dimensions (*) must 
                       be smaller or equal to those of the 'feature' tensor.
        cluster_assignment: A tensor indicating the cluster assignment of each extra sequence.
                            Shape: (N_extra,)
        cluster_assignment_count: A tensor containing the number of extra 
                                 sequences assigned to each cluster center.
                                 Shape: (N_clust,)

    Returns:
        A tensor containing the average feature representation for each cluster. 
        Shape: (N_clust, N_res, *) 
    """
    N_clust, N_res = feature.shape[:2]
    N_extra = extra_feature.shape[0]

    ######################################################################
    # TODO:
    # 1. **Prepare for Accumulation:**
    #     * Broadcast the `cluster_assignment` tensor to have the same shape as `extra_feature`.
    #     This is necessary for compatibility with `torch.scatter_add`.
    # 2. **Accumulate Features:**
    #     * Use `torch.scatter_add` to efficiently sum (or accumulate) the `extra_feature` values  for each cluster.  The broadcasted `cluster_assignment` tensor will define the grouping. 
    # 3. **Calculate Averages:**
    #     * Divide the accumulated features by the `cluster_assignment_count` + 1 to obtain the average feature representations for each cluster. 
    ######################################################################


    unsqueezed_extra_shape = (N_extra,) + (1,) * (extra_feature.dim()-1)
    unsqueezed_cluster_shape = (N_clust,) + (1,) * (feature.dim()-1)

    cluster_assignment = cluster_assignment.view(unsqueezed_extra_shape).broadcast_to(extra_feature.shape)
    cluster_sum = torch.scatter_add(feature, dim=0, index=cluster_assignment, src=extra_feature)
    cluster_assignment_count = cluster_assignment_count.view(unsqueezed_cluster_shape).broadcast_to(feature.shape)
    cluster_average = cluster_sum / (cluster_assignment_count + 1)

    # End of your code

    return cluster_average



def summarize_clusters(features):
    """
    Calculates cluster summaries by applying cluster averaging to the MSA amino acid 
    representations and deletion counts.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Modifies:
        The 'features' dictionary in-place by adding the following:
            * cluster_deletion_mean: Average deletion counts for each cluster center, 
                                     scaled for numerical stability.
            * cluster_profile: Average amino acid representations for each cluster center.
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    ######################################################################
    # TODO:
    # 1. **Calculate Cluster Deletion Means:**
    #     * Employ the `cluster_average` function to calculate the average deletion counts for each cluster using  'msa_deletion_count' and related features from the 'features' dictionary. 
    #     * Apply the transformation `2/torch.pi * torch.arctan(x/3)` to the computed cluster deletion means to map them between -1 and 1. 
    # 2. **Calculate Cluster Profiles:**
    #      * Use the `cluster_average` function again to calculate the average amino acid representations for each cluster, using 'msa_aatype' and its corresponding 'extra' feature from the 'features' dictionary.
    ######################################################################


    cluster_deletion_mean = cluster_average(
        features['msa_deletion_count'],
        features['extra_msa_deletion_count'],
        features['cluster_assignment'],
        features['cluster_assignment_counts'] 
    )

    cluster_deletion_mean = 2/torch.pi * torch.arctan(cluster_deletion_mean/3)
    cluster_profile = cluster_average(
        features['msa_aatype'],
        features['extra_msa_aatype'],
        features['cluster_assignment'],
        features['cluster_assignment_counts']
    )

    features['cluster_deletion_mean'] = cluster_deletion_mean
    features['cluster_profile'] = cluster_profile

    # End of your codes

    return features

def crop_extra_msa(features, max_extra_msa_count=5120, seed=None):
    """
    Reduces the number of extra sequences in the MSA to a fixed size for computational efficiency.

    Args:
        features: A dictionary containing feature representations of the MSA.
        max_extra_msa_count: The maximum number of extra sequences to retain.
        seed: An optional integer seed for the random number generator. 
              Use this to ensure reproducibility.

    Modifies:
        The  'features' dictionary in-place by cropping the following keys to include
        only the first 'max_extra_msa_count' sequences:
            * Any key starting with 'extra_' 
    """

    N_extra = features['extra_msa_aatype'].shape[0]
    gen = torch.Generator(features['extra_msa_aatype'].device)
    if seed is not None:
        gen.manual_seed(seed)

    max_extra_msa_count = min(max_extra_msa_count, N_extra)

    ######################################################################
    # TODO:
    # 1. **Generate Random Permutation:**
    #     * Use `torch.randperm(N_extra)` with the provided generator (`gen`) to create a random ordering of the extra sequence indices.
    # 2. **Select Subset:**
    #     * Slice the random permutation to select the first `max_extra_msa_count` indices.
    # 3. **Crop Features:**
    #     * Iterate through the `features` dictionary.  For each key that starts with  'extra_', slice the corresponding value using the selected indices to retain only the first `max_extra_msa_count` sequences.  
    ######################################################################


    inds_to_select = torch.randperm(N_extra, generator=gen)[:max_extra_msa_count]
    for k, v in features.items():
        if k.startswith('extra_'):
            features[k] = features[k][inds_to_select]

    # End of your code

    return features

def calculate_msa_feat(features):
    """
    Prepares the final MSA feature representation for protein structure prediction.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_clust, N_res, 49) representing the final MSA features,
        formed by concatenating processed cluster information and deletion-related values. 
    """
    
    N_clust, N_res = features['msa_aatype'].shape[:2]
    msa_feat = None

    ######################################################################
    # TODO:
    # 1. **Prepare Features:**
    #     * Obtain the following features from the 'features' dictionary:
    #        - 'cluster_msa' (Shape: (N_clust, N_res, 23))
    #        - 'msa_deletion_count' (Shape: (N_clust, N_res))
    #        - 'cluster_deletion_mean' (Shape: (N_clust, N_res))
    #        - 'cluster_profile' (Shape: (N_clust, N_res, 23))
    #     * Calculate:
    #         - `cluster_has_deletion`: Boolean tensor of shape (N_clust, N_res, 1) indicating the presence of deletions.
    #         - `cluster_deletion_value`: Scaled deletion count with shape (N_clust, N_res, 1).
    # 2. **Concatenate Features:** 
    #     * Use `torch.cat` to concatenate the following tensors, in this order, along the last dimension to create the final 'msa_feat' tensor: 
    #          - `cluster_msa` 
    #          - `cluster_has_deletion`
    #          - `cluster_deletion_value`
    #          - `cluster_profile`
    #          - `cluster_deletion_mean`
    ######################################################################


    cluster_msa = features['msa_aatype']

    cluster_has_deletion = (features['msa_deletion_count'] > 0).float().unsqueeze(-1)

    cluster_deletion_value = 2/torch.pi * torch.arctan(features['msa_deletion_count'] / 3)
    cluster_deletion_value = cluster_deletion_value.unsqueeze(-1)

    cluster_deletion_mean = features['cluster_deletion_mean'].unsqueeze(-1)
    cluster_profile = features['cluster_profile']

    msa_feat = torch.cat((cluster_msa, cluster_has_deletion, cluster_deletion_value, cluster_profile, cluster_deletion_mean), dim=-1)

    # End of your code

    return msa_feat

def calculate_extra_msa_feat(features):
    """
    Prepares the extra MSA feature representation for protein structure prediction. 
    This function is similar to 'calculate_msa_feat' but operates on  extra MSA sequences
    and includes padding to account for potential masked tokens. 

    Args:
        features: A dictionary containing feature representations of the MSA.

    Returns:
        A tensor of shape (N_extra, N_res, 25) representing the final extra MSA features.
    """

    N_extra, N_res = features['extra_msa_aatype'].shape[:2]
    extra_msa_feat = None

    ######################################################################
    # TODO:
    # 1. **Prepare Features:**
    #     * Obtain the following features from the 'features' dictionary:
    #        - 'extra_msa_aatype' (Shape: (N_extra, N_res, 22))
    #        - 'extra_msa_deletion_count' (Shape: (N_extra, N_res))
    #     * Calculate:
    #         - `extra_msa_has_deletion`: Boolean tensor of shape (N_extra, N_res, 1) indicating the presence of deletions.
    #         - `extra_msa_deletion_value`: Scaled deletion count with shape (N_extra, N_res, 1).
    #      * Create a zero padding tensor of shape (N_extra, N_res, 1).
    # 2. **Pad and Concatenate Features:** 
    #     * Concatenate the `extra_msa_aatype` and zero padding along the last dimension.
    #     * Use `torch.cat` to concatenate the following tensors, in this order, along the last dimension to create the final 'extra_msa_feat' tensor: 
    #          - `extra_msa` (with padding)
    #          - `extra_msa_has_deletion`
    #          - `extra_msa_deletion_value`
    ######################################################################


    padding = torch.zeros((N_extra, N_res, 1))
    extra_msa = torch.cat((features['extra_msa_aatype'], padding), dim=-1)
    extra_msa_has_deletion = (features['extra_msa_deletion_count'] > 0 ).float().unsqueeze(-1)
    extra_msa_deletion_value = 2/torch.pi * torch.arctan(features['extra_msa_deletion_count']/3)
    extra_msa_deletion_value = extra_msa_deletion_value.unsqueeze(-1)

    extra_msa_feat = torch.cat((extra_msa, extra_msa_has_deletion, extra_msa_deletion_value), dim=-1)

    # End of your code

    return extra_msa_feat

if __name__=='__main__':
    batch = create_features_from_a3m('alignments_hhr/test_tautomerase/test_tautomerase.a3m')
    basic_features = torch.load('test_outputs/basic_features.pt', map_location='cpu')
    extra_msa_feat = torch.load('test_outputs/extra_msa_feat.pt', map_location='cpu')

    print(batch.keys())

    print('Target feat:')
    print((batch['target_feat']-basic_features['target_feat'][...,1:, 0]).abs().max())
    print('Residue index:')
    print((batch['residue_index']-basic_features['residue_index'][...,0]).float().abs().max())
    print('MSA feat:')
    print((batch['msa_feat']-basic_features['msa_feat'][...,0]).abs().max())
    print('Extra MSA feat:')
    print((batch['extra_msa_feat']-extra_msa_feat).abs().max())

    