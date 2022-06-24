from itertools import product
from pm import Algo, Order, Miner, Smooth, DriftDiscoverMode

log_name = 'bpic2018Insp'
Miner.generate_csv(log_name)
for algo, cut, top in product(Algo, [500], [None]):
    print(f'{algo.name},{cut},{top}')
    miner = Miner(log_name, Order.FRQ, algo, cut, top, update=True,
                  drift_discover_algorithm=DriftDiscoverMode.ADWIN, no_sampling=True, summarization=True)
    try:
        miner.process_stream()
        miner.save_results()
        miner.generate_summary(log_name)
    except Exception as e:
        print(f'\tError: {e}')
