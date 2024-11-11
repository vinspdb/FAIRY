import subprocess
from platform import system
from sys import stdout
from enum import Enum
from collections import Counter
from datetime import timedelta
from os import path, makedirs, listdir

import numpy as np
import pandas
from collections import deque
from matplotlib import pyplot
from numpy.f2py.crackfortran import verbose
from pandas import DataFrame, read_csv
from time import process_time
from tempfile import TemporaryDirectory
from pm4py import read_bpmn, read_pnml
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.streaming.importer.csv import importer as csv_stream_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
#from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from river.drift import ADWIN
#from sympy.abc import alpha
import pm4py
import time
from datetime import datetime
from dl import Abstractor
pandas.options.mode.chained_assignment = None  # default='warn'

CASE_ID_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
FINAL_ACTIVITY = '_END_'


class Algo(Enum):
    IND = 1
    SPL = 2
    ILP = 3



class Order(Enum):
    FRQ = 1
    MIN = 2
    MAX = 3


class Smooth(Enum):
    SMA = 1
    EMA = 2
    SLW = 3


class DriftDiscoverMode(Enum):
    NORMAL = 1
    ADWIN = 2


class Miner:

    @staticmethod
    def generate_csv(log_name, case_id=CASE_ID_KEY, activity=ACTIVITY_KEY, timestamp='time:timestamp'):
        """
        Converte il file XES in input in uno stream di eventi ordinati cronologicamente con formato CSV. Ogni traccia
        viene estesa con un evento conclusivo che ne definisca il termine
        :param log_name: nome del file XES (eventualmente compresso) contenente il log di eventi
        :param case_id: attributo identificativo dell'istanza di processo (con aggiunta del prefisso 'case:')
        :param activity: attributo identificativo dell'attività eseguita
        :param timestamp: attributo indicante l'istante di esecuzione di un evento
        """
        csv_path = path.join('eventlog', 'CSV', log_name + '.csv')
        if not path.isfile(csv_path):
            print('Generating CSV file from XES log...')
            xes_path = path.join('eventlog', 'XES', log_name)
            xes_path += '.xes.gz' if path.isfile(xes_path + '.xes.gz') else '.xes'
            log = xes_importer.apply(xes_path, variant=xes_importer.Variants.LINE_BY_LINE)
            for trace in log:
                trace.append({activity: FINAL_ACTIVITY, timestamp: trace[-1][timestamp] + timedelta(seconds=1)})
            dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
            dataframe = dataframe.filter(items=[activity, timestamp, case_id]).sort_values(timestamp, kind='mergesort')
            dataframe = dataframe.rename(columns={activity: ACTIVITY_KEY, case_id: CASE_ID_KEY})
            makedirs(path.dirname(csv_path), exist_ok=True)
            dataframe.to_csv(csv_path, index=False)

    def __init__(self, log_name, order, algo, cut, top, filtering=False, frequency=False, update=False, smooth=None,
                 trace_window_size=0, drift_discover_algorithm=DriftDiscoverMode.NORMAL, no_sampling=False,
                 filter_treshold=None, parallelism_threshold=0.1, summarization=False):
        """
        Metodo costruttore
        :param log_name: nome del file CSV contenente lo stream di eventi
        :param order: criterio di ordinamento delle varianti
        :param algo: algoritmo da utilizzare nell'apprendimento del modello
        :param cut: numero di istanze da esaminare preliminarmente
        :param top: numero di varianti da impiegare nella costruzione del modello (None per la distribuzione di Pareto)
        :param filtering: booleano per l'utilizzo di tecniche di filtering
        :param frequency: booleano per l'utilizzo delle frequenze nella costruzione del modello
        :param update: booleano per l'apprendimento dinamico del modello
        :param smooth: criterio di smoothing dei dati
        :param trace_window_size: dimensione finestra per sliding window o numero di istogrammi su cui fare la media
        mobile
        :param drift_discover_algorithm: approccio utilizzato per la scoperta dei concept drift
        :param no_sampling: booleano che indica se evitare o utilizzare il sampling dei log per la generazione dei\
        :param filter_treshold: float compreso tra 0 e 1 che rappresenta valore filtro
        :param parallelism_threshold: float compreso tra 0 e 1 che indica il valore di parallelism da utilizzare su
        algoritmo SPL
        """
        self.log_name = log_name
        self.order = order
        self.algo = algo
        self.cut = cut
        self.top = top
        self.filtering = filtering
        self.frequency = frequency
        self.update = update
        self.processed_traces = 0
        self.variants = Counter()
        self.best_variants = None
        self.models = []
        self.drift_moments = []
        self.drift_variants = []
        self.evaluations = []
        #self.new_model_counter = 0
        self.training_time = 0
        self.smoothed_freq_dictionary = Counter()
        self.smooth = smooth
        self.trace_window_size = trace_window_size
        self.window_queue = None
        self.adwin = None
        self.drift_discover_algorithm = drift_discover_algorithm
        self.no_sampling = no_sampling
        self.filter_treshold = filter_treshold
        self.parallelism_threshold = parallelism_threshold
        self.summarization = summarization
        self.abstractor = None
        self.model_update = False
        self.cont_model = 0
        self.current_time = 0

    def time_format(self, time_stamp):
        '''
        :param time_stamp: oggetto timestamp
        :return: converte l'oggetto timestamp utile in fase di calcolo dei tempi
        '''
        try:
            date_format_str = '%Y-%m-%d %H:%M:%S.%f%z'
            conversion = datetime.strptime(time_stamp, date_format_str)
        except:
            date_format_str = '%Y-%m-%d %H:%M:%S%f%z'
            conversion = datetime.strptime(time_stamp, date_format_str)
        return conversion
    def process_stream(self):
        """
        Processa iterativamente uno stream di eventi in formato CSV, ignorando attività che si ripetano in modo
        consecutivo per un numero di occorrenze superiore a due e aggiornando il contatore delle varianti in
        corrispondenza di un evento finale. Dopo aver esaminato un dato numero di istanze preliminari, viene generato
        un modello di processo. Tale modello sarà valutato su ciascuna delle istanze successive
        """
        print('Processing event stream...')
        stream = csv_stream_importer.apply(path.join('eventlog', 'CSV', self.log_name + '.csv'))
        traces = {}
        start = process_time()
        self.init_window_queue()
        self.no_sampling_mode_init()
        self.set_default_filter_value()
        for event in stream:
            case = event[CASE_ID_KEY]
            activity = event[ACTIVITY_KEY]
            if self.model_update == True:
                start_time = self.time_format(self.last_time_event)
                end_time = self.time_format(event['time:timestamp'])
                diff = end_time - start_time
                self.current_time = 86400 * diff.days + diff.seconds + diff.microseconds / 1000000


            if activity == FINAL_ACTIVITY:
                new_trace = tuple(traces.pop(case))
                self.variants[new_trace] += 1
                self.processed_traces += 1
                if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
                    self.window_queue.append(new_trace)
                if self.processed_traces > self.cut - self.trace_window_size and self.drift_discover_algorithm == \
                        DriftDiscoverMode.NORMAL:
                    self.build_variants_dictionary(new_trace)
                if self.processed_traces == self.cut:
                    if self.summarization:
                        self.abstractor = Abstractor(self.variants, self.log_name, self.calculate_pareto_denominator())
                        #self.abstractor.build_neural_network_model()
                        self.abstractor.find_hyperparameters()
                    self.select_best_variants()
                    self.learn_model(self.best_variants) if not self.summarization else \
                        self.learn_model(self.abstractor.build_summary(self.variants))
                    end = process_time()
                    self.evaluations.append([None, None, None, end - start])
                    start = end
                    if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
                        self.adwin_window_init()
                elif self.processed_traces > self.cut:
                    stdout.write(f'\rCurrent model: {len(self.models)}\tCurrent trace: {self.processed_traces}')
                    self.evaluations.append(self.evaluate_model(new_trace))
                    if self.update:
                        self.discover_concept_drift(event)
                    end = process_time()
                    self.evaluations[-1].append(end - start)
                    start = end
            elif case not in traces:
                traces[case] = [activity]
            elif len(traces[case]) == 1 or traces[case][-1] != activity or traces[case][-2] != activity:
                traces[case].append(activity)

    def set_default_filter_value(self):
        """
        Imposta il valore di filtro di default in base all'algoritmo utilizzato
        """
        if self.filter_treshold is None and self.filtering:
            if self.algo == Algo.IND:
                self.filter_treshold = 0.2
            elif self.algo == Algo.SPL:
                self.filter_treshold = 0.4
            else:
                self.filter_treshold = 0.25

    def init_window_queue(self):
        """
        Inizializza la coda utilizzata dalla sliding window su trace, sliding window su istogrammi e dal metodo
        ADWIN. Settando anche la dimensione massima opportuna.
        """
        if self.smooth == Smooth.SLW or self.smooth == Smooth.SMA:
            self.window_queue = deque(maxlen=self.trace_window_size)
        elif self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
            self.window_queue = deque()

    def no_sampling_mode_init(self):
        """
        Avvalora filtering e frequency a true se il sistema viene inizializzato in modo da non utilizzare il sampling
        dei log
        """
        if self.no_sampling:
            self.filtering = True
            self.frequency = True

    def discover_concept_drift(self, event):
        """
        Richiama il metodo preposto alla scoperta dei concept drift, in base all' approccio definito in fase di
        inizializzazione.
        """
        if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
            self.concept_drift_finder_with_adwin(self.evaluations[-1][2], event)
        else:
            self.select_best_variants()
            if self.best_variants.keys() != self.drift_variants[-1].keys():
                if not self.summarization:
                    self.learn_model(self.best_variants)
                else:
                    self.abstractor.update_training_set(self.variants, self.calculate_pareto_denominator())
                    start_time = time.process_time()
                    self.abstractor.build_neural_network_model()
                    end_time = time.process_time()
                    self.last_time_event = event['time:timestamp']
                    self.training_time = end_time - start_time
                    self.model_update = True
                    self.learn_model(self.abstractor.build_summary(self.variants))

    def build_variants_dictionary(self, new_trace):
        """
        Avvalora dizionario istogrammi secondo il criterio di smoothing scelto
        :param new_trace: ultimo trace letto
        """
        if self.processed_traces < self.cut and (self.smooth == Smooth.EMA or self.smooth is None):
            return
        self.build_sliding_window_dictionary(new_trace) if self.smooth == Smooth.SLW else self.build_sma_window_queue()\
            if self.smooth == Smooth.SMA else self.build_ema_smoothed_dictionary() if self.smooth == Smooth.EMA \
            else None

    def select_best_variants(self):
        """
        Determina le varianti più significative all'istante corrente secondo il criterio d'ordine selezionato
        """
        top_variants = self.top
        if self.drift_discover_algorithm == DriftDiscoverMode.NORMAL:
            variants = self.build_usable_ema_dictionary() if self.smooth == Smooth.EMA \
                else (self.smoothed_freq_dictionary if self.smooth == Smooth.SLW else
                      (self.build_usable_sma_dictionary() if self.smooth == Smooth.SMA else self.variants))
        else:
            variants = self.variants
        variants = {k: v for k, v in sorted(variants.items(), key=lambda x: x[1], reverse=True)}
        if not self.no_sampling:
            denominator = self.calculate_pareto_denominator()
            if top_variants is None:
                counter = 0
                top_variants = 0
                while counter / denominator < 0.8:
                    counter += list(variants.values())[top_variants]
                    top_variants += 1
        else:
            top_variants = len(variants.keys())
        if self.order == Order.FRQ:
            self.best_variants = {item[0]: item[1] for item in list(variants.items())[:top_variants]}
        else:
            candidate_variants = list(item[0] for item in variants)
            candidate_variants.sort(key=lambda v: len(v), reverse=self.order == Order.MAX)
            self.best_variants = {var: variants[var] for var in candidate_variants[:top_variants]}

    def calculate_pareto_denominator(self):
        """
        Definisce il denominatore da utilizzare per il calcolo di Pareto secondo il criterio di smoothing selezionato
        """
        if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
            return len(self.window_queue)
        elif self.smooth == Smooth.SMA:
            return 1
        else:
            return self.processed_traces if self.smooth is None or self.smooth == Smooth.EMA else \
                self.trace_window_size if self.smooth == Smooth.SLW and self.processed_traces >= self.trace_window_size\
                else self.processed_traces

    def build_ema_smoothed_dictionary(self):
        """
        Costruisce dizionario variante-array con in posizione 0 frequenza rilevata, 1 frequenza smussata
        """
        for variant in self.variants.keys():
            if variant not in self.smoothed_freq_dictionary:
                self.smoothed_freq_dictionary[variant] = np.array([self.variants[variant],
                                                                   float(self.variants[variant])])
            elif self.variants[variant] != self.smoothed_freq_dictionary[variant][0]:
                self.smoothed_freq_dictionary[variant][1] = 0.5 * self.smoothed_freq_dictionary[variant][1] + 0.5 * \
                                                            self.variants[variant]
                self.smoothed_freq_dictionary[variant][0] = self.variants[variant]

    def build_usable_ema_dictionary(self):
        """
        Converte dizionario variante-array frequenze in dizionario variante-frequenza smussata in modo da poter essere
        utilizzato all'interno della funzione select_best_variants()
        :return: smoothed_dictionary: dizionario con chiave Variante e valore EMA frequenze
        """
        smoothed_dictionary = Counter()
        for variant in self.smoothed_freq_dictionary.keys():
            smoothed_dictionary[variant] = self.smoothed_freq_dictionary[variant][1]
        return smoothed_dictionary

    def build_sliding_window_dictionary(self, new_trace):
        """
        Costruisce dizionario variante-frequenza, relativa alla presenza della variante all'interno della
        window considerata.
        :param new_trace: trace corrente rilevato dal dataset
        """
        if len(self.window_queue) >= self.trace_window_size:
            old_trace = self.window_queue.popleft()
            if old_trace == new_trace:
                self.window_queue.append(new_trace)
                return
            self.smoothed_freq_dictionary[old_trace] -= 1
            self.smoothed_freq_dictionary.pop(old_trace) if self.smoothed_freq_dictionary[old_trace] == 0 else None
        self.window_queue.append(new_trace)
        self.smoothed_freq_dictionary[new_trace] += 1

    def build_sma_window_queue(self):
        """
        Costruisce dizionario variante-frequenza relativa all'istogramma corrente, e lo incoda
        """
        current_histogram = {}
        for variant in self.variants:
            current_histogram[variant] = self.variants[variant] / sum(self.variants.values())
        self.window_queue.append(current_histogram)

    def build_usable_sma_dictionary(self):
        """
        Costruisce dizionario variante- media delle frequenze relativa alla variante rilevata negli istogrammi presenti
        nella coda.
        :return: Dizionario con chiave Variante e valore media calcolata con SMA
        """
        smoothed_dictionary = Counter()
        for variant in self.variants.keys():
            for current_dict in self.window_queue:
                if variant in current_dict.keys():
                    smoothed_dictionary[variant] += current_dict[variant] / self.trace_window_size
        return smoothed_dictionary

    def learn_model(self, variants_histogram):
        """
        Genera un modello di processo utilizzando le varianti più significative all'istante corrente
        """
        log = EventLog()
        for variant, occurrence in variants_histogram.items():
            if not self.smooth == Smooth.SMA:
                for i in range(int(occurrence) if self.frequency else 1):
                    log.append(Trace({ACTIVITY_KEY: activity} for activity in variant))
            else:
                log.append(Trace({ACTIVITY_KEY: activity} for activity in variant))
        if self.algo == Algo.IND:
            model = pm4py.discovery.discover_petri_net_inductive(log, noise_threshold=0.2)
        elif self.algo == Algo.ILP:
            model = pm4py.discovery.discover_petri_net_ilp(log, alpha=0.25)
            #model = pm4py.discovery.discover_petri_net_inductive(log, noise_threshold=0.1)


        else:
            with TemporaryDirectory() as temp:
                log_path = path.join(temp, 'log.xes')
                variant = xes_exporter.Variants.LINE_BY_LINE
                xes_exporter.apply(log, log_path, variant, {variant.value.Parameters.SHOW_PROGRESS_BAR: False})
                model_path = path.join(temp, 'model.bpmn' if self.algo == Algo.SPL else 'model.pnml')
                script = path.join('scripts', 'run.bat' if system() == "Windows" else 'run.sh')
                args = (script, self.algo.name, str(self.filtering), log_path, path.splitext(model_path)[0],
                        str(self.filter_treshold), str(self.parallelism_threshold))
                subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                model = bpmn_converter.apply(read_bpmn(model_path)) if self.algo == Algo.SPL else read_pnml(model_path)
        self.models.append(model)
        #self.new_model_counter +=1
        self.drift_moments.append(self.processed_traces)
        self.drift_variants.append(self.best_variants)


    def get_prev_model(self, log):
        i = -1
        while True:
            try:
                fitness = pm4py.fitness_alignments(log, *self.models[i], multi_processing=False)['average_trace_fitness']
                # variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
                # parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: False}
                #precision = pm4py.precision_alignments(log, *self.models[i], multi_processing=True)
                variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
                parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: False}
                # precision = pm4py.precision_alignments(log, *self.models[-1], multi_processing=False, parameters=parameters)
                precision = precision_evaluator.apply(log, *self.models[i], variant=variant, parameters=parameters)
                return fitness, precision
                break
            except:
                i = i - 1
    def evaluate_model(self, trace):
        """
        Valuta il modello di processo sull'istanza fornita in input
        :param trace: istanza di processo da impiegare nella valutazione
        :return results: lista contenente in posizione 0 la fitness, 1 la precision, 2 la f_measure
        """
        log = EventLog([Trace({ACTIVITY_KEY: activity} for activity in trace)])
        #variant = fitness_evaluator.Variants.ALIGNMENT_BASED

        if self.model_update==True:
            if self.current_time<self.training_time:
                fitness, precision = self.get_prev_model(log)
            else:
                self.model_update = False
                try:
                    fitness = pm4py.fitness_alignments(log, *self.models[-1], multi_processing=False)[
                        'average_trace_fitness']
                    variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
                    parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: False}
                    # precision = pm4py.precision_alignments(log, *self.models[-1], multi_processing=False, parameters=parameters)
                    precision = precision_evaluator.apply(log, *self.models[-1], variant=variant, parameters=parameters)
                except:
                    fitness, precision = self.get_prev_model(log)
        else:
            try:
                fitness = pm4py.fitness_alignments(log, *self.models[-1], multi_processing=False)['average_trace_fitness']
                variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
                parameters = {variant.value.Parameters.SHOW_PROGRESS_BAR: False}
                #precision = pm4py.precision_alignments(log, *self.models[-1], multi_processing=False, parameters=parameters)
                precision = precision_evaluator.apply(log, *self.models[-1], variant=variant, parameters=parameters)

            except:
                fitness, precision = self.get_prev_model(log)

        f_measure = 2 * fitness * precision / (fitness + precision) if fitness != 0 else 0
        return [fitness, precision, f_measure]

    def adwin_window_init(self):
        """
        Inizializza la finestra utilizzata dall' oggetto ADWIN con i trace utilizzati per la generazione del primo
        modello.
        """
        self.adwin = ADWIN()
        for trace in self.window_queue:
            self.adwin.update(self.evaluate_model(trace)[2])

    def concept_drift_finder_with_adwin(self, f_measure, event):
        """
        La funzione si occupa di passare la f-measure calcolata all' algoritmo ADWIN e se quest' ultimo, dopo aver
        aggiunto il nuovo valore rileva un concept drift, vengono utilizzati i trace della finestra rilevata da ADWIN
        su cui il concept drift si è verificato per la generazione di un nuovo modello.
        :param f_measure: f-measure calcolata sull' ultimo trace letto.
        """
        in_drift, in_warning = self.adwin.update(f_measure)
        if in_drift:
            number_of_traces_to_be_removed = len(self.window_queue) - int(self.adwin.width)
            for i in range(number_of_traces_to_be_removed):
                old_trace = self.window_queue.popleft()
                self.variants[old_trace] -= 1
                self.variants.pop(old_trace) if self.variants[old_trace] == 0 else None
            if self.summarization:
                self.abstractor.update_training_set(self.variants, self.calculate_pareto_denominator())
                #self.abstractor.build_neural_network_model()
                start_time = time.process_time()
                self.abstractor.build_neural_network_model()
                end_time = time.process_time()
                self.last_time_event = event['time:timestamp']
                self.training_time = end_time - start_time
                self.model_update = True


            self.select_best_variants()
            self.learn_model(self.best_variants) if not self.summarization else \
                self.learn_model(self.abstractor.build_summary(self.variants))

    def find_best_filter(self):
        """
        Trova il valore per il filtro (e per il parallelism per SPL) in grado di massimizzare la f-measure media sul 10%
        del dataset utilizzando tale filtro per la generazione del modello
        """
        current_filter = 0
        current_parallelism = 0.1 if self.algo == Algo.SPL else 1
        best_filter = 0
        best_fmeasure = 0
        best_parallelism = 0.1 if self.algo == Algo.SPL else None
        while current_parallelism <= 1:
            print(f'Current parallelism: {current_parallelism}')
            while current_filter <= 1:
                self.filter_treshold = current_filter
                self.parallelism_threshold = current_parallelism if self.algo == Algo.SPL else None
                self.learn_model(self.best_variants) if not self.summarization else \
                    self.learn_model(self.abstractor.build_summary(self.variants))
                current_fmeasure = 0
                for trace in self.window_queue:
                    try:
                        current_fmeasure += self.evaluate_model(trace)[2]
                    except Exception:
                        break
                print(f'with filter: {current_filter} fmeasure: {current_fmeasure / len(self.window_queue) }')
                if current_fmeasure > best_fmeasure:
                    best_filter = current_filter
                    best_fmeasure = current_fmeasure
                    best_parallelism = current_parallelism if self.algo == Algo.SPL else None
                current_filter += 0.02
            current_parallelism += 0.1
            current_filter = 0 if self.algo == Algo.SPL else None
        print(f'Best filter value: {best_filter} with f-measure: {best_fmeasure / len(self.window_queue)}')
        print(f'Best parallelism: {best_parallelism}') if self.algo == Algo.SPL else None

    def compute_model_complexity(self, index):
        """
        Calcola la complessità del modello indicizzato
        :param index: indice del modello di cui calcolare la complessità
        :return: numero di posti, numero di transizioni, numero di archi e metrica "Extended Cardoso" del modello
        """
        net = self.models[index][0]
        ext_card = 0
        for place in net.places:
            successor_places = set()
            for place_arc in place.out_arcs:
                successors = frozenset(transition_arc.target for transition_arc in place_arc.target.out_arcs)
                successor_places.add(successors)
            ext_card += len(successor_places)
        return len(net.places), len(net.transitions), len(net.arcs), ext_card

    def save_results(self):
        """
        Esporta report, valutazioni e modelli di processo
        """
        print('\nExporting results...')
        filtering = 'UFL' if self.filtering else 'NFL'
        frequency = 'UFR' if self.frequency else 'NFR'
        update = 'D' if self.update else 'S'
        top_variants = 'P' if self.top is None else self.top
        file = f'{self.order.name}.{self.algo.name}.{self.cut}.{top_variants}.{filtering}.{frequency}.{update}'
        folder = path.join('results', self.log_name, 'report')
        makedirs(folder, exist_ok=True)
        top_variants = max(len(variants.keys()) for variants in self.drift_variants)
        columns = ['trace', 'places', 'transitions', 'arcs', 'ext_cardoso',
                   *[f'trace_{i}' for i in range(1, top_variants + 1)]]
        report = DataFrame(columns=columns)
        report.index.name = 'n°_training'
        for index, current_variants in enumerate(self.drift_variants):
            traces = [f'[{v}]{k}' if self.order == Order.FRQ else f'[{len(k)}:{v}]{k}' for k, v in
                      current_variants.items()] + [None] * (top_variants - len(current_variants))
            report.loc[len(report)] = [self.drift_moments[index], *self.compute_model_complexity(index), *traces]
        report.to_csv(path.join(folder, file + '.csv'))
        folder = path.join('results', self.log_name, 'evaluation')
        makedirs(folder, exist_ok=True)
        columns = ['fitness', 'precision', 'f-measure', 'time']
        evaluation = DataFrame(self.evaluations, columns=columns)
        evaluation.index.name = 'n°_evaluation'
        total_time = evaluation['time'].sum()
        evaluation.loc['AVG'] = evaluation.mean()
        evaluation.loc['TOT'] = [None, None, None, total_time]
        evaluation.to_csv(path.join(folder, file + '.csv'))
        folder = path.join('results', self.log_name, 'petri')
        makedirs(folder, exist_ok=True)
        #for index, model in enumerate(self.models):
        #    model_info = f'-{index}' if self.update else ''
        #    pnml_exporter.apply(model[0], model[1], path.join(folder, file + model_info + '.pnml'), model[2])
        #    pn_visualizer.save(pn_visualizer.apply(*model), path.join(folder, file + model_info + '.png'))

    def save_variant_histogram(self, y_log=False):
        """
        Esporta l'istogramma delle varianti di processo
        :param y_log: booleano per l'utilizzo di una scala logaritmica sull'asse delle ordinate
        """
        file = ('frequency' if self.order == Order.FRQ else 'length') + '_histogram.png'
        folder = path.join('results', self.log_name)
        makedirs(folder, exist_ok=True)
        if not path.isfile(path.join(folder, file)):
            y_axis = self.variants.values() if self.order == Order.FRQ else [len(v) for v in self.variants.keys()]
            pyplot.bar(range(1, len(self.variants) + 1), y_axis)
            pyplot.title(f'Traces processed: {self.processed_traces}     Variants found: {len(self.variants)}\n')
            pyplot.xlabel('Variants')
            pyplot.ylabel('Frequency' if self.order == Order.FRQ else 'Length')
            if y_log:
                pyplot.semilogy()
            pyplot.savefig(path.join(folder, file))

    @staticmethod
    def generate_summary(log_name):
        """
        Genera una visualizzazione sintetica dei risultati ottenuti
        :param log_name: nome del log per il quale generare un sommario dei risultati
        """
        folder = path.join('results', log_name)
        if not path.isdir(folder):
            print('No results found')
            return
        print('Generating summary...')
        error_file = path.join(folder, 'errors.csv')
        errors = read_csv(error_file) if path.isfile(error_file) else DataFrame(columns=['algo'])
        columns = ['order', 'top-variants', 'set-up', 'fitness', 'precision', 'f-measure', 'time', 'ext_cardoso']
        for algo in Algo:
            summary = DataFrame(columns=columns)
            for error in errors.loc[errors['algo'] == algo.name].itertuples():
                row = [error.order, error.top, f'{error.filtering} {error.frequency} {error.update}'] + ['-'] * 5
                summary.loc[len(summary)] = row
            for file in listdir(path.join(folder, 'evaluation')):
                if algo.name in file:
                    parameters = file.split(sep='.')
                    row = [parameters[0], parameters[3], f'{parameters[4]} {parameters[5]} {parameters[6]}']
                    evaluation = read_csv(path.join(folder, 'evaluation', file), dtype={'n°_evaluation': 'str'})
                    row.extend(evaluation[val][len(evaluation) - 2] for val in ('fitness', 'precision', 'f-measure'))
                    row.append(evaluation['time'][len(evaluation) - 1])
                    report = read_csv(path.join(folder, 'report', file))
                    row.append(str(report['ext_cardoso'].tolist()))
                    summary.loc[len(summary)] = row
            summary['top-variants'] = summary['top-variants'].replace('P', -1).astype(int)
            summary = summary.sort_values(['order', 'top-variants', 'set-up'], ignore_index=True)
            summary['top-variants'] = summary['top-variants'].replace(-1, 'P')
            summary.index.name = 'experiment'
            summary.index += 1
            summary.to_csv(path.join(folder, algo.name + '.csv'))
