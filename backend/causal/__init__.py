"""
EconoCausal - Causal Inference Package.
Ensures networkx / dowhy compatibility across recent versions.
"""

import networkx as nx

# Compatibility bridge for NetworkX >= 3.2 where d_separated was renamed to is_d_separator
try:
    import networkx.algorithms.d_separation as ds
    if not hasattr(nx.algorithms, "d_separated"):
        nx.algorithms.d_separated = ds.is_d_separator
    if not hasattr(nx, "d_separated"):
        nx.d_separated = ds.is_d_separator
except Exception:
    pass
