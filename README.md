# ConAction

<p align="center">
	<img src="imgs/trinity_of_covariation.png" alt='Instantiations of the Trinity of Covariation' width="300"/>
</p>

<p align="center">
	<a href="https://github.com/ellerbrock/open-source-badges/" target="_blank">
		<img alt="Open Source Love" src="https://badges.frapsoft.com/os/v1/open-source.png?v=103">
	</a>
	<a href="https://conaction.readthedocs.io/en/latest/?badge=latest" target="_blank">
		<img alt="Documentation Status" src="https://readthedocs.org/projects/conaction/badge/?version=latest">
	</a>
	<a href="https://badge.fury.io/py/conaction" target="_blank">
		<img alt="PyPI version" src="https://badge.fury.io/py/conaction.svg">
	</a>
	<a href="https://img.shields.io/pypi/dm/conaction" target="_blank">
		<img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/conaction">
	</a>
	<a href="https://github.com/galenseilis/ConAction/blob/main/LICENSE" target="_blank">
		<img alt="License" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg">
	</a>
	<a href="https://github.com/psf/black" target="_blank">
		<img alt="Code style: Black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
	</a>
</p>


## Project Description
The ConAction library provides mathematical functions that are inspired from the metaphor of the Trinity of Covariation as part of the MSc thesis of Galen Seilis.

Supervisory Committee Members

- Supervisor: [Dr. Edward Dobrowolski](https://www2.unbc.ca/people/dobrowolski-dr-edward) ([Department of Mathematics and Statistics, UNBC](https://www2.unbc.ca/math-statistics))
- Committee Member: [Dr. Brent Murray](https://web.unbc.ca/~murrayb/) ([Department of Biology, UNBC](https://www2.unbc.ca/biology))
- Committee Member: [Dr. Mohammad El Smaily](https://smaily.opened.ca/) ([Department of Mathematics and Statistics, UNBC](https://www2.unbc.ca/math-statistics))
- External Examiner: [Dr. Anne Condon](https://www.cs.ubc.ca/~condon/) ([Department of Computer Science](https://www.cs.ubc.ca/about))

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

## Citation

```

@mastersthesis{seilisthesis2022,
  author  = "Galen Seilis",
  title   = "ConAction: Efficient Implementations and Applications of Functions Inspired by the Trinity of Covariation",
  school  = "University of Northern British Columbia",
  year    = "2022",
  address = "3333 University Way, Prince George, British Columbia, V2N 4Z9, Canada",
  month   = "September",
  doi = 10.24124/2022/59312,
  url = https://doi.org/10.24124/2022/59312
}
```

![Star History Chart](https://api.star-history.com/svg?repos=galenseilis/ConAction&type=Date)
