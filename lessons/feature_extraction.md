# Feature Extraction

Hi everyone and welcome to this video on feature extraction in AlphaFold. I'm Kilian Mandon and this is the fourth video in our series where we'll be implementing AlphaFold from scratch. It's also the first one where we are going into AlphaFold specific content. So, if you're already familiar with machine learning and attention, you're good to start here, just know that you'll either need to implement the MultiHeadAttention module in the assignment for the last video, or copy our version from github, to go through with the AlphaFold implementation yourself.

Feature extraction means converting the domain specific data formats we want to use as inputs into tensors, the data format for machine learning. There are two non-obvious questions here. The first one is what that domain specific data should be, that is which biological data actually carries information on the protein structure. And the second is how we transform it into tensors. So, let's get into it!

<!-- This video and the next two will be a little shorter than the introductory ones. There's not so much theory to explain, and it's more of a list of steps you need to implement. It's still quite a bit of work to actually implement this in python, as you'll see when working through the notebook. Since feature extraction has to be flexible regarding its input, you need a bit more non-standard PyTorch code to handle it. But for now, let's start with the theory. -->

AlphaFold uses three types of inputs. 
The  first is the one you'd most likely expect: The amino acid sequence of the protein you want to predict the structure of. It is given as a string and each letter represents one of the 20 amino acids. 
But AlphaFold uses two additional inputs that try to get information on the structure from the protein's evolutionary history.
The first of these two is MSA data, which is short for multiple sequence alignment. It is a list of protein sequences found in other organisms that are highly similar to the target sequence, which means they probably originate from the same ancestor. 
And the last is the 3D structure of so called templates, proteins that are very similar to the target and where the structure has already been determined. Historically, this has been the most relevant data for structure prediction. Interestingly enough, AlphaFold doesn't really seem to need the template stack as input, in particular if the MSA is diverse and provides rich evolutionary information. We won't implement the template stack in AlphaFold to make the code easier to follow, and it might be less used than you think anyway. In the popular online tool ColabFold for example, the template stack is disabled by default as well.

After construction of the inputs, AlphaFold feeds them through the input embedder and the Evoformer, as shown here. Without the template stack, the input pipeline looks like this. These two parts, input embedding and the evoformer stack, are what we'll do in the next two videos. Today, we'll construct the four tensors on the left. 

Two of them we can directly check off. Residue_index is nothing but the range from 0 to r-1, to be used for position encoding. And target_feat is simply a one-hot encoding of the target amino acid sequence, using 21 tokens for the 20 amino acids and an additional "unknown" token. Construction of the two MSA features is a little more complicated.

We'll start with a quick introduction to what sequence alignments are. Proteins are sequences of amino acids. We can extract a lot of information from the evolutionary history of proteins. For example, we can check if a region is highly conserved over different proteins or if it's less so and changes a lot. That might tell us if the sequence is closer to the center and interacts with many residues to produce the 3D structure, or if its a loose loop. Another possible analysis is too look for residues that only switch together, so called Co-Evolution. Since mutation is random, we'd generally expect the residues to change independently of each other. But evolution often sorts out mutations that show a loss of function. That means that, if two residues often co-evolve, they might interact with each other and if one of them is substituted, the other one needs to undergo substitution as well to form a stable protein.
Explicit calculations like these were commonly done in the past. But you can imagine that there are way more, deeper correlations within biological sequences. For that reason, there's been a paradigm shift in sequence analysis. Instead of relying on manually crafted algorithms for specific tasks, we now train general statistical models and machine learning algorithms on large datasets. These models are capable of learning complex patterns and making predictions directly from the data, offering a more flexible and powerful approach to understanding genomics. And this is also how we'll handle it in AlphaFold. 

Knowing that the evolutionary history might be helpful for structure prediction, our first step is to find sequences that are similar to our target sequence.

Problem: Proteins can mutate by substitution, insertion or deletion, meaning we can't just compare positions pointwise. An introduction of a new residue would misalign a large part of the sequence. This is why we do "alignment", where we try to estimate where insertions and deletions happened. This is done by the Needlemann-Wunsch algorithm. It assigns scores to correctly aligned amino acids, wrongly aligned ones (where the penalty depends on the type of substitution) and gaps in the alignment. Finding the optimal alignment can be solved quite elegantly using a method called Dynamic Programming, but we won't go into the details here. 

The big problem is that the Sequence Databases are vast. To keep it in memory, you can expect to need about 70 Gigabytes of memory, more than what typical hardware can offer. This is a big part of the computational cost of running AlphaFold. In this series, we won't compute the alignments ourselves, but use precomputed ones. The alignment file we use for testing the implementation was generated by the ColabFold notebook, you get the file together with the structure prediction when running ColabFold. The ColabFold notebook itself runs on a free-to-use google machine, like all Colab Notebooks, and that hasn't the specs to do sequence alignment either. The ColabFold Notebook queries a public server to compute the alignments. You can do so as well for a limited number of sequences, but for large numbers, you would need to think about setting up your own hardware.

Either way, we end off with a .a3m file containing the sequence alignment data. It looks like this: 
You can see that there's little metadata in the file, because it was generated solely from the sequence given to ColabFold. It consists of alternating lines starting with a ">" and some scores, followed by a sequence. The first sequence is the query sequence. It is from a Tautomerase from E. coli, and its also the default sequence when you open up ColabFold.

The sequences immediately below it consist only of letters, which means they don't contain insertions or deletions, only substitutions. The second sequence has 50.8% identity to the target, and the third one has 40.6% identity. 
In line 23, we've got a insertion for the first time. It's represented by the dash at the end of the sequence. This means that, compared to this sequence, the target sequence had an Amino Acid inserted at this position.
Further down in line 393, we have our first deletion: The amino acids glycine, glutamine and glycine, represented by the lower-case characters gqg, are present in this sequence but not in the target sequence. 
Note that when talking about insertions and deletions here, we can't really tell if these truly were insertions or deletions because we don't know the evolutionary history. If our target protein was older than the homolog sequence, it might be the other way around. By Insertions and Deletions we mean the terms as starting with the homolog sequence, then going to our target.

Basically, AlphaFold uses two pieces of information from the MSA. First, the types of amino acids at each position, for some of the sequences individually as one-hot encodings and for others as averages over a group. Second, the positions and number of deletions in the sequences.

For the latter, we can go through the sequences, remove all lowercase characters and count how many we removed before each amino acid. Note that after we did so, all sequences have the same length as the target sequence. AlphaFold only uses sequences that are unique after removing deletions. This means that if we have two sequences in the alignment that only vary in their deletions, we only keep the first. After removing the deletions, all sequences are one-hot encoded using 22 classes: The 20 amino acids and the unknown and gap tokens. The very first sequence, which is the target sequence, is additionally one-hot encoded using 21 classes, not including the gap token, to create the aatype feature.

The one-hot encoded sequences are of shape (N_seq, N_res, 22). We can additionally calculate the distribution over the different amino acids at each position by calculating the mean over the different sequences. This distribution is used later to resample some of the amino acids.

The next mechanism in AlphaFold is the selection of cluster centers. We randomly select a number of sequences as cluster centers, always including the target sequence as the first center. All other sequences are assigned to the cluster center that they are most similar to. For each of the cluster centers, we extract the deletions and amino acids individually, for the other sequences in the cluster we average them. The cluster centers aren't chosen to be specifically well distributed. We simply generate a random permutation of the index range (1, N_seq), prepend the index 0 to always include the target sequence first, then select the first 512 of these as cluster centers, while we gather the rest as Extra MSAs.

Before the extra sequences are assigned to the cluster centers, the cluster centers are randomly modified to increase the robustness of the model. Note that this regularization isn't only used in training but also during inference, which is why we need to implement it as well. This process is called masking and it involves the following steps:
1) With a probability of 15%, each position in each cluster center is selected for potentially being substituted. For all selected positions there is
2) a 10% chance of being replaced with a uniformly sampled random amino acid
3) a 10% chance of being replaced with an amino acid sampled from the MSA profile at its position
4) a 10% chance of not being replaced (this might also happen by the previous steps)        
5) a 70% chance of being replaced with a special token (masked_msa_token)
This is easy enough using normal python, in PyTorch we have to think a little how we can achieve this using tensor operations. The basic idea is to gather all these replacement distributions in a single distribution for each position in each cluster center, then sampling from that using torch.distributions.Categorical. 
For example, the category of uniform sampling would be a distribution of [1/20, ..., 1/20, 0, 0, 0], with zeros for the unknown token, the gap token and the mask token. We can calculate these distributions for the individual replacement pathways, scale them by the chance for the path (i.e. 10% for uniform distribution) and add them up. After sampling from the distribution, we create a mask with probability 15% and replace the residues with the sampled ones where the mask says so. The notebook will guide you through the exact implementation.

After masking the cluster centers, the extra sequences are assigned to the clusters. The assignment simply counts how many residues in the extra sequence agree with the cluster center. This is called the hamming distance of the two sequences. Note that we only count agreement between amino acids, agreement of gaps doesn't count. Each extra sequence is assigned to the cluster center it agrees with mostly.

Now that we understand the concept of clustering and masking, we can take a look at the individual tensors we want to create as features for AlphaFold.

The feature aatype is a one-hot encoding of the input sequence.

cluster_msa is the same for all sequences that were selected as cluster centers. Note that we need to additional tokens in the one-hot encoding, for gap tokens and mask tokens.

The feature cluster_has_deletion is one for every residue in the cluster centers that had a deletion on its left, zero otherwise.

cluster_deletion_value actually counts the number of deletions left to each residue, then normalizes it by 2/pi*arctan(d/3) to the range (-1, 1), which is better suited as a network input.

The features extra_msa, extra_msa_has_deletion and extra_msa_deletion_value are identical to the ones for the cluster centers, but calculated for all sequences that were not selected. These features will be used as input through the less complex, memory-friendly Extra MSA Stack.

For the main input, the extra sequences only contribute as averages for each cluster, by the features cluster_deletion_mean and cluster_profile. They contain just what the names suggest: cluster_deletion_mean is the average number of deletions left to each residue for each sequence in the clusters, normalized to the range [-1,1] using arctan again, and cluster_profile is a distribution over amino acids at each position. Note that the averages also include the cluster centers.

After the calculation of the individual features, the last thing we do is to concatenate some of them to get the final inputs:

The feature 'target_feat' is the 'aatype' feature.
The feature 'residue_index' is a range of [0, ..., N_res-1], to be used for positional encodings.
The feature 'msa_feat' is constructed by concatenating 'cluster_msa', 'cluster_has_deletion', 'cluster_deletion_value', 'cluster_deletion_mean' and 'cluster_profile'.
The feature 'extra_msa'feat' is constructed by concatenating 'extra_msa', 'extra_msa_has_deletion' and 'extra_msa_deletion_value'.

Note that there's some randomness in the input creation, notably the selection of the cluster centers and during masking. As we'll see later, AlphaFold does its full prediction multiple times, recycling the predicted positions and other outputs in the newer passes. The inputs for the model are created for each run individually by just repeating all these steps, so they are somewhat different due to the randomness involved.


So, this is how feature extraction works for AlphaFold. 

Selecting the relevant input data for and shaping it into tensors are key for using machine learning in new problem settings.  I think it's really cool to see an example of how we can do this for a problem like protein structure prediction. You can find the tutorial notebook for this topic Linked in the description, where you'll build the full feature extraction pipeline for AlphaFold by yourself. 

In the next two videos we'll explore how the Evoformer, AlphaFold's core module, transforms these features. Then we'll tackle the inverse of today's topic: Getting from tensors to actual protein structures. So, see you in the next videos, and happy coding!