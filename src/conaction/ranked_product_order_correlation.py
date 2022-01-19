'''
This is a script illustrating a proof
of concept. It is yet another way of
formulating multivariate association.
This has some interesting properties. For one
thing, it relies on measuring a sort of simultanous
'monotonicity'. While I need to give more thought
as to whether "monotonicity" is exactly the right
term, the point is to measure how much variables
go up or down together within assuming a specific
model structure or distribution. It may also be
relatively outlier-resistant because it uses something
like a rank.


Imagine we have a data matrix
of shape m by n.

Each row of the data matrix is a row vector.

A product order can therefore be created on
the rows of the data matrix.

https://en.wikipedia.org/wiki/Product_order

A 'pseudorank' can be assigned to each row
vector to be the number of row vectors that
it was strictly less than under the product
order. I'm calling them "pseudoranks" for now,
but I should examine if they could also be
called "graded posets" or "ranked posets".

https://en.wikipedia.org/wiki/Graded_poset
https://en.wikipedia.org/wiki/Ranked_poset

This pseudorank gives a multivariate description
of some of the structure in the data.

If a histogram is made of the pseudoranks, a
couple of patterns can be observed.

One pattern is a perfectly uniform distribution
of pseudoranks. This would imply that the exists
a sort of total order in the data.

https://en.wikipedia.org/wiki/Total_order

A second pattern that might occur is clumping of
pseudoranks near zero, or a power-law-shaped
distribution. This would mean that many of the
data points are not comparable under this relation.

The following code loads the iris dataset, and calculates
the pseudoranks as described above.

Note that I create a matrix P, and then sum over it after.
This is because visualizing P as an image can show useful
information about the data. For example, there is a "blocky"
pattern in the image that happens to correspond to the target
classes of irises.

If the image of P were to appear as an upper triangle matrix,
then the rows of X are a total order. I'm not sure that the
converse has to be true yet. There may be a linear algebra
approach to sensibly create a distance (metric) function
between the total order and the observed data.

Another approach for quantifying how close the data
matches a total order is to calculate the information
entropy of the distribution of pseudoranks. The closer
to uniform the distribution, the closer the data is to
a partial order. The closer to a uniform distribution,
the higher the information entropy. This suggests that
higher information entropy in the pseudoranks will
quantify the extent to which the data is totally ordered.

https://en.wikipedia.org/wiki/Entropy_(information_theory)

Since the entropy will depend on the number of bins in the
distribution, which will be discrete because the number of rows
and comparisons is discrete. Thus the entropy can simply be normalized
by the maximal entropy for that event space: H_max = \frac{1}{m} \log(m).
Bootstrapping the rows would allow for a distribution of these entropy
scores for which a hypothesis test could readily be devised to test
if a given dataset has a higher entropy (as described above) than if there
no order in the rows. Such a hypothesis testing approach would allow for
the comparison of whether two datasets had difference amounts of 'structure'
in the sense of their product order as previously described.

Note that the pseudoranks are paired with the rows, and thus each object
of measurement is paired with a pseudorank. Given two collections of
variables that were measured on the same objects of measurement, we can
calculate a correlation coefficient on their pseudoranks. While having the
same objects of study is not always a convenient constraint, it allows for the
estimation of whether two collections of variables have similar product-orders.

It may or may not be useful to plot the Hasse diagrams of the partial orders
for visual examination. These can be quite 'busy looking' for larger datasets,
but may allow for a visual sanity check that there is order.

Given a collection of variables, we can take a computational approach to
exploring structure in the dataset. Exempting the empty set, we can calculate
the entropy association score on each set of variables in the powerset of
variables. While there is more thinking to be done, at this exact moment I
suspect that these scores will not be independent. A motivating example of this
can be given. Let's say we have the collection of variables {X, Y, Z}, and that
we know {X, Y} score low, {X,Z} scored low, and {Y, Z} scored high. With these
hypothetical premises, could I predict if {X,Y,Z} would score high or low? My 
suspicion is that it would score low.

For computational complexity, I suspect it will be something like O(n m^2) because
each row is compared to each other row (hence m^2), and there will be n components
to compare in each row.

Challenge: Why not rank the magnitudes or dot products of the row vectors instead?
Answer: The idea of measuring the monotonicty will not be preserved by ranking on
those functions of the data. The magnitude would treat vectors with very negative
components as higher rank than those closer to the zero vector. The dot product
could have a mix of components that are extremely negative or positive that could
cancel out in the summation to a number close to zero.

Simulation Idea: Use simulated annealing to simultaneously maximize pearson's R
, or my product-moment correlation coefficient, and minimize this monotonicty
correlation. This might illustrate something about the importance of noise if
they can be very different.

Sanity Check: I found that if the data matrix X was occupied with independently and
identically distributed uniform random variables, that all entries had a pseudorank
of zero. This suggests a lack of strong functional relationship between variables
will lead to low entropy scores (as described above), and that the noise is unlikely
to create an ordering by chance alone.

Sanity Check: I found the same result of all the pseudoranks being zero if the
entries in the data were IID standard normal distributions.

Note: A low entropy score over a collection of variables doesn't mean that there
are not relationships.

I wonder how this relates to the notion of comonotonicty.
https://en.wikipedia.org/wiki/Comonotonicity
https://www.sciencedirect.com/science/article/pii/S0047259X09001353
http://homepages.ulb.ac.be/~grdeelst/DJV.pdf
https://www.researchgate.net/publication/4958878_The_Concept_of_Comonotonicity_in_Actuarial_Science_And_Finance_Theory
https://www.researchgate.net/publication/237300089_Risk_Measures_and_Comonotonicity_A_Review
https://arxiv.org/pdf/2002.12278.pdf
'''

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from numba import jit
import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv('file:///home/galen/Dropbox/UNBC/Rader/AllCombinedRNASeqData.csv')

    
def permute_columns(x):
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]

def grade_entropy(X):
    P = np.zeros((X.shape[0], X.shape[0]))
    @jit(nopython=True, fastmath=True) # does fastmath actually help?
    def grade(q):
        v = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            if np.all(X[j] < X[q]):
                v[j] += 1
        return v
    for i in tqdm.tqdm(range(X.shape[0])):
        P[:, i] = grade(i) 
    bins, counts = np.unique(np.sum(P, axis=0), return_counts=True)
    probs = counts / np.sum(counts)
    entropy = np.sum(probs * np.log(probs)) / (np.log(1/np.sum(counts)))
    return entropy

def statistic_permute(X, stat_func=grade_entropy, iters=100):
    y = []
    for r in tqdm.tqdm(range(iters), total=iters):
        y.append(stat_func(permute_columns(X)))
    return y

X = np.array(df[df.keys()[1:]]).T
ref_val = np.mean(X)
print(ref_val)
s = np.array(statistic_permute(X, stat_func=np.mean, iters=1000))
print(np.mean(s >= ref_val))
