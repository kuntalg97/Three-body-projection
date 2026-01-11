# Three-body-projection
This generates an effective composite potential by projecting three-body interactions onto a simple, pairwise potential.

An initial coarse-grained (CG) simulation using three-body potentials (e.g., Stillinger–Weber) should be conducted, along with a separate force-matching run using simple pairwise potentials. This code projects the former onto the latter, effectively generating a composite potential that implicitly captures three-body effects. This approach can be further refined through iterative force-matching.

Relevant papers: [J. Chem. Phys. 159, 224105 (2023)](https://doi.org/10.1063/5.0176716) and [J. Chem. Phys. 154, 044104 (2021)](https://doi.org/10.1063/5.0026651)

(Codes for computing spatial correlation functions: radial and angular distribution functions are also provided here.)
