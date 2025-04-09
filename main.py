from pm import Algo, Order, Miner,DriftDiscoverMode
import sys
log_name = sys.argv[1]
init = int(sys.argv[2])
update = sys.argv[3]
drift_mode = sys.argv[4]
summ = sys.argv[5]
method = sys.argv[6]
result_name = sys.argv[7]
Miner.generate_csv(log_name)

if update == 'True':
    upd = True
else:
    upd= False

if drift_mode == 'ADWIN':
    print('drift ADWIN')
    drift_mode = DriftDiscoverMode.ADWIN
elif drift_mode == 'HDDM_W':
    print('drift HDDM_W')
    drift_mode = DriftDiscoverMode.HDDM_W
elif drift_mode == 'EDDM':
    print('drift EDDM')
    drift_mode = DriftDiscoverMode.EDDM
else:
    drift_mode = DriftDiscoverMode.NORMAL

if summ == 'True':
    gen_sum = True
else:
    gen_sum = False


print('Configuration-->',log_name,init,update,drift_mode)
for a in [Algo.IND]:
    miner = Miner(
        log_name=log_name,
        order=Order.FRQ,
        algo=a,
        cut=init,
        top=None,
        method=method,
        result_name = result_name,
        update=upd,
        frequency=True,
        filtering=True,
        drift_discover_algorithm=drift_mode,
        summarization=gen_sum
    )
    miner.process_stream()
    miner.save_results()
    miner.generate_summary(log_name, result_name)
