from pm import Algo, Order, Miner,DriftDiscoverMode
import sys
log_name = sys.argv[1]
init = int(sys.argv[2])
update = sys.argv[3]
drift_mode = sys.argv[4]
summ = sys.argv[5]
Miner.generate_csv(log_name)

if update == 'True':
    upd = True
else:
    upd= False

if drift_mode == 'adwin':
    drift = DriftDiscoverMode.ADWIN
else:
    drift = DriftDiscoverMode.NORMAL

if summ == 'True':
    gen_sum = True
else:
    gen_sum = False

if alg == 'im':
    disc_alg = Algo.IND
elif alg == 'spl':
    disc_alg = Algo.SPL
elif alg == 'ilp':
    disc_alg = Algo.ILP

print('Configuration-->',log_name,init,update,drift)

miner = Miner(
        log_name=log_name,
        order=Order.FRQ,
        algo=disc_alg,
        cut=init,
        top=None,
        update=upd,
        frequency=True,
        filtering=True,
        drift_discover_algorithm=drift,
        summarization=gen_sum
    )
miner.process_stream()
miner.save_results()
miner.generate_summary(log_name)
