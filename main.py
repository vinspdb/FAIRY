from pm import Algo, Order, Miner,DriftDiscoverMode

log_name = 'RequestForPayment' #event log name
init = 650# number of trace for the first model
Miner.generate_csv(log_name)

for a in Algo:
    miner = Miner(
        log_name=log_name,
        order=Order.FRQ,
        algo=a,
        cut=init,
        top=None,
        update=True,
        frequency=True,
        filtering=True,
        drift_discover_algorithm=DriftDiscoverMode.ADWIN,
        summarization=False
    )

    miner.process_stream()
    miner.save_results()
    miner.generate_summary(log_name)
