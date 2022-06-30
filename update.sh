 # Python Packages
pip install --upgrade pip
pip install --upgrade build
pip install --upgrade twine
pip install --upgrade bump2version
pip install --upgrade twine

# Current Package
python3 -m twine upload dist/*
