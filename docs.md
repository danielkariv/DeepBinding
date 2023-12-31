# Docs
Contains links and data I collected, to have some bases to work on the project. Will be expend when I find more stuff.

# The datasets are:
All files extract to main directory, like this: 'RBNS_testing/...', 'RBNS_training/...', 'RNCMPT_training', 'RNAcompete_sequences.txt'.
RBPX_Ynm.seq - is a DNA samples, with counts of how much seen (was told it isn't that important for the model). notice that it isn't fully representing of the connections, as it is a small sample. Should be used to try a model that can detect features. Main ways to do so are with unsupervised learning models, one of them is AutoEncoder, which we can split and take only the encoder to use later.
RNAcompete_sequences.txt - RNA seqs, it is the input of the main model that predicts the binding strength. Used by all RBPs to test binding.
RBPX.txt - (RNA-Binding-Protein = RBP) is lines of floats, and should be the measured binding. It related to RNAcompete_sequences. This is what the model should predict in the end.

# The input of training:
we got 16 RBPs + RNCMPT sets with RBNS experiments samples.

# Some knowledge from the project.pdf:
Binding sites (BSs) of a particular RBP share a common pattern called 'motif'. 
In the example, it is 10 letters long, and can be anywhere in the RNA seq. 
In a later one, it is 6 latters long. 
It seems we can represent motifs as a position weight matrix (PWM) which each position has weights for the 4 possible letters.
RNAcompete suppose to be a RNA pool with all possible 9-mers (9 letters long motfis?), and it suppose to detect RBP binding to specific 9-mers?.
31-41nt variable region, all 9-mers each at least 16 times, ~240k probes (all in the RNAcompete_sequences.txt). 
RNA bind-n-seq - RBNS files. has different concentrations, and seqs for each that it detects.

# Computational chanllenge:
input = RBNS data (4-6 seq files) of one RBP, and a list of RNAcompete probes (1 seq file).
goal/output = predict RNA binding intensity for each RNAcompete probe (run on the probes in the file, and get a float value).

# The overall idea I got:
basically, we use the RBPX_input,5nm,20nm...1300nm to train a model. This model is later used to estimate binding values and validated using the given the RNAcompete probes list and RNAcompete_sequences (only for validation).
We probably do have to train a second model that uses the first model to estimate the values more closely to the real ones, and then we can estimate the unknown values (what we sumbit later) with a trained model using the specific known RBNS.
basically, Model 1 is features extracting, learn on specific RBP, but Model 2 that use Model 1, is also learn but on all the known RBPs binding scores so we have a scoring model for unknown RBPs biding scores.

# Some stuff about keras:
I using something called TensorBoard, which runs a server under http://localhost:6006
It gives graphs of different values while training models.

# Info,Data,Video,Etc:
A paper about this project, by Eitamar Tripto, supervised by Dr. Yaron Orenstein.
https://in.bgu.ac.il/en/robotics/thesis/TriptoEitamar19912.pdf

DeepSelex more info:
https://academic.oup.com/bioinformatics/article/36/Supplement_2/i634/6055905
https://github.com/OrensteinLab/DeepSELEX

A paper that gave me the idea of how to model our system, though I implemented something much simlper than them but we can use it to see where to progress next.
https://arxiv.org/pdf/1906.03087.pdf
Something about scores on very similar dataset if not the same based on it's size (31 RBPs with RNAcompete)
Didn't view it yet! it is locked but maybe it can be openned with University Wifi.
https://academic.oup.com/bib/article-abstract/22/6/bbab149/6278600?redirectedFrom=fulltext&login=false

Some other stuff by Yaron Orenstein
The video he talked about the project on Module (start at around the middle point)
https://lemida.biu.ac.il/blocks/video/viewvideo.php?id=164583&courseid=81067&type=2
Video from BGU in English. He shows some ideas how to load the data and something similar to our project (Part 1).
https://www.youtube.com/watch?v=_ccpjpWeJXw
A second video is about motif finding, it uses PBM and finds PWM (Part 2).
https://www.youtube.com/watch?v=jL-SKPi9fbY

Brand new paper I saw, Any ReLU network can be represent using a 3-layer network.
Didn't read yet but could mean we can avoid some deep and hard to run models, with much better performance, equaled networks.
https://www.arxiv.org/abs/2306.11827



# Solution:
Sliding window for k-mers (k=6?) -> model (finds motifs) -> output is the score (label). Sum them up and have the final score.
Yaron said that we can start by taking input.seq as negative score, and 1300nM.seq as positive.
Later with RBP1.txt, we can compare to the scores we receives (using RNAcompete_sequences.txt).

Sliding window can be maybe be done with Generators (too big data, needs to happen on fly, per batch).
model, questionable how to design, it suppose to detect the motifs using the BnS and then we will use it to score the RNAcompete_sequences.txt. Basically making a model for each RBP.
the scoring itself, doesn't seems to be learnt, and should be somewhat simple. using maybe sliding windows on the RNAcompete, we could score it like that?

# Some debug information:
Copied from the script: (some files don't neccerally appear when running, need to handle it)
 NOTE: ('x', 'x', 'x', '+', '+', 'x') works the best on most RBPs, expect of those with more than 6 classes(files), there is one of it on 16 RBPs. or mayabe: [('x', 'x', 'x', '+', '+', '+')] (From what I see, it is worse than just two plus. )
all_combinations = [('-', '-', '-', '-', '+', '+', '+', '+', '+', '+'), # 0.0831
                    ('-', '-', '-', '-', '-', '+', '+', '+', '+', '+'), # 0.1150
                    ('-', '-', '-', '-', '-', '-', '+', '+', '+', '+'), # 0.1395 365-9800
                    ('-', '-', '-', '-', '-', '+', '+', '+', '+', 'x'), # 0.0612
                    ('-', '-', '-', '-', 'x', 'x', '+', '+', '+', 'x'), # 0.1092
                    ('-', '-', '-', '-', 'x', 'x', '+', '+', '+', 'x'), # 0.1092
                    ('x', 'x', 'x', 'x', 'x', 'x', '+', '+', '+', 'x'), # 0.1092
                    ('x', 'x', 'x', 'x', 'x', 'x', '+', '+', '+', '+'), # 0.1092
                    ('x', 'x', 'x', 'x', 'x', '+', '+', '+', '+', '+'), # 0.1395 121-9800
                    ('x', 'x', 'x', 'x', 'x', '+', '+', '+', 'x', 'x'), # 0.1150
                    ]# itertools.product(operations, repeat=num_inputs)

# What left to try:
Maybe try looking into the labels of the data. Right now it just based on teh file the sequence comes from, but maybe we could add the number of appearence into the mix. Questionable how to do it, as we do softmax, the max value can be 1.0.