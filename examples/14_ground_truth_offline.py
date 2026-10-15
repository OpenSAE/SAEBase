"""Pure synthetic known factor controls without pretrained model downloads."""
from saebase.reliability.ground_truth import make_factors,fit_matching,evaluate_matching
from saebase.reliability.interaction_truth import exact_order_energy
from saebase.reliability.controls import check_coordinates

def main():
    fixture=make_factors("superposition",seed=17)
    mapping=fit_matching(fixture.calibration,fixture.calibration)
    print("Held-out recovery:",evaluate_matching(fixture.test,fixture.test,mapping))
    print("Third-order interaction energy:",exact_order_energy("third_order"))
    print("Invariance controls:",check_coordinates(seed=10))
if __name__=="__main__":
    main()
