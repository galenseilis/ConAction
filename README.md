# ConAction

<img src="imgs/trinity_of_covariation.png" alt='Instantiations of the Trinity of Covariation' width="300"/>

## Project Description
The ConAction library provides mathematical functions that are inspired from the metaphor of the Trinity of Covariation as part of the MSc thesis of Galen Seilis.

Supervisory Committee Members

- Supervisor: [Dr. Edward Dobrowolski](https://www2.unbc.ca/people/dobrowolski-dr-edward) ([Department of Mathematics and Statistics, UNBC](https://www2.unbc.ca/math-statistics))
- Committee Member: [Dr. Brent Murray](https://web.unbc.ca/~murrayb/) ([Department of Biology, UNBC](https://www2.unbc.ca/biology))
- Committee Member: [Dr. Mohammad El Smaily](https://smaily.opened.ca/) ([Department of Mathematics and Statistics, UNBC](https://www2.unbc.ca/math-statistics))
- External Examiner: TBA

The code in this repository is intended to support researchers analyzing multivariate data.

The thesis provides an extensive background reading for this package, and can be found at (link needed).

## Installation
ConAction is available through [PyPi](https://pypi.org/project/conaction/), and can be installed via `pip` using
```
pip install conaction
```
or 
```
pip3 install conaction
```

## Example Usage

```python
from conaction import estimators
import numpy as np

X = np.random.normal(size=1000).reshape((100,10)) # Get a 100 x 10 data table

estimators.pearson_correlation(X) # Compute the 10-linear Pearson correlation coefficient
```

## Documentation


Build documentation locally:

```bash
cd /path/to/conaction/docs
make html
```

## Tutorials

Coming soon.

## Publications

Coming soon.

## License

BSD 3-Clause License

Copyright (c) 2021, Galen Seilis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
