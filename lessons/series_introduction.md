Hi there, and welcome to AlphaFold Decoded. This series is designed to guide you through every step of implementing AlphaFold 2 from scratch.

I'm a biology student, and I was fascinated in my first semester that machine learning could actually predict the 3 dimensional structure of proteins, but even more so by the fact that this was a really recent development. 

I had some background in machine learning - mostly in computer vision - and i wanted to understand how AlphaFold works, but I found that to be quite hard. Basically, the only material that covered the architecture with enough detail for doing an implementation yourself were the paper by deepmind and a number of open source implementations. But if you tried before, these can be really hard to read.

<!-- Cut -->

In other disciplines like computer vision or natural language processing, it's common to have "from-scratch" guides, like the transformer tutorial by Andrej Karpathy. This is supposed to be somewhat like that, but for AlphaFold. What makes this tricky is that AlphaFold is quite a bit more complex then transformers. Handling structural information is inherently more difficult than text because you need proper mathematical models to describe it, and learning that takes some time.

<!-- Cut, next can be read out -->

This series is structured in nine-parts, and its designed so that you can go through with it without any particular prerequisites, except general python knowledge. In the first three videos, we'll go through a full introduction to the most important parts of machine learning, with an introduction to tensors, the basic building block of ML, an introduction to machine learning in general, and the attention mechanism. After that, we start with the AlphaFold specific content: Feature Extraction, the Evoformer, Feature Embedding, a video on advanced 3D geometry, the Structure Module and finally, putting all the parts together.

<!-- Cut -->

That's quite a bit of content. If you think you're already familiar with some of the non-alphafold specific content, you can of course skip over that. But still, it’s important to know that this journey will require a significant time investment. Most of the videos are about 20 minutes long, and somewhat dense. Each of the videos is accopanied by a jupyter notebook, and depending on your experience in programming, you can expect to spend about 4 to 6 hours on each - maybe even more. I still think this is worth it - learning computational biology could really change your impact on the future of biology - but you'll need a certain amount of commitment. Of course, you don't need to decide that right now. You can just start the series and see how it fits for you. In any case, I'm excited to go on this journey with you and to hear what you think. So, see you in the next video!