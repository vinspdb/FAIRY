from collections import Counter

import numpy as np
from sklearn import preprocessing

from tensorflow.keras.layers import Embedding, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
from time import perf_counter
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

FINAL_ACTIVITY_SYMBOL = '_END_'

space = {'cellsize': scope.int(hp.loguniform('cellsize', np.log(10), np.log(150))),
         'dropout': hp.uniform("dropout", 0, 1),
         'batch_size': hp.choice('batch_size', [7, 8, 9, 10]),
         'num_heads': hp.choice('num_heads', [4, 8, 16]),  # Attention heads
         'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
         'num_transformer_blocks': hp.quniform('num_transformer_blocks', 1, 3, 1),
         }

class Abstractor:

    def __init__(self, histogram, dataset_name, trace_window_size, examples_size=4):
        """
        Metodo costruttore.
        :param histogram: istogramma trace-frequenza da utilizzare per addestrare ed utilizzare la rete neurale
        :param dataset_name: nome dataset da cui i trace sono stati ricavati
        :param trace_window_size: dimensione finestra di trace considerati per le varie operazioni
        :param examples_size: dimensione dell'array utilizzato per rappresentare gli esempi all'interno del training set
        """
        print('TRANSFORMERS')
        self.mapping_dictionary = {}
        self.trace_histogram = histogram
        self.variant_seeds = []
        self.examples_size = examples_size
        self.dataset_name = dataset_name
        self.trace_window_size = trace_window_size
        self.map_dictionary_init()
        self.trace_max_lenght = self.set_max_trace_size()
        self.x_training, self.y_training = self.build_training_set(examples_size)
        seed = 123
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.le = preprocessing.LabelEncoder()
        self.y_training = self.le.fit_transform(self.y_training)
        self.best_score = np.inf
        self.best_model = None
        self.best_time = 0
        self.best_numparameters = 0

    def map_dictionary_init(self):
        """
        Inizializza il dizionario activity- valore corrispondete con chiave il simbolo finale di trace e valore 1
        """
        if not self.mapping_dictionary:
            self.mapping_dictionary[FINAL_ACTIVITY_SYMBOL] = 1

    def get_activity_map_value(self, activity):
        """
        Inserisce la activity come chiave nel dizionario assegnandole un valore univoco se non presente
        :param activity: activity da inserire nel dizionario e di cui restituire il valore corrispondente
        :return: intero che rappresenta univocamente l'activity all'interno del dizionario
        """
        if activity not in self.mapping_dictionary.keys():
            self.mapping_dictionary[activity] = len(self.mapping_dictionary.keys()) + 1
        return self.mapping_dictionary[activity]

    def map_trace(self, trace):
        """
        Converte trace in array di interi, dove ogni intero rappresenta il valore mappato della corrispondente activity.
        Aggiunge il valore corrispondente al simbolo di terminazione del trace in ultima posizione
        :param trace: trace da convertire
        :return: array di interi corrispondente al trace passato
        """
        mapped_trace = np.empty(len(trace) + 1, dtype=int)
        for i in range(0, mapped_trace.size - 1, 1):
            mapped_trace[i] = self.get_activity_map_value(trace[i])
        mapped_trace[-1] = self.mapping_dictionary[FINAL_ACTIVITY_SYMBOL]
        return mapped_trace

    def build_training_set(self, examples_size):
        """
        Costruisce il training set per la rete come sequenza di array della dimensione passata in input. Gli array
        contengono interi, diversi per ogni activity. L'intero nella posizione finale rappresenta l'etichetta dell'
        esempio
        :param examples_size: dimensione array esempi
        :return training set generato da istogramma
        """
        x_training = []
        y_training = []
        for trace in self.trace_histogram:
            i = 0
            current_mapped_trace = self.map_trace(trace)
            while i < self.trace_histogram[trace]:
                j = 1
                while j < current_mapped_trace.size:
                    current_example = np.zeros(examples_size)
                    values = current_mapped_trace[0:j] if j <= examples_size else \
                        current_mapped_trace[j - examples_size:j]
                    y_training.append(current_mapped_trace[j])
                    current_example[examples_size - values.size:] = values
                    x_training.append(current_example)
                    j += 1
                i += 1
        return x_training, y_training

    def update_training_set(self, trace_histogram, trace_window_size):
        """
        Aggiorna istogramma e parametri da utilizzare per la rete neurale
        :param trace_histogram: istogramma trace-frequenza
        :param trace_window_size: Dimensione finestra trace utilizzata per costruire istogramma
        """
        self.trace_histogram = trace_histogram
        self.trace_window_size = trace_window_size
        self.trace_max_lenght = self.set_max_trace_size()
        self.x_training, self.y_training = self.build_training_set(self.examples_size)
        self.y_training = self.le.fit_transform(self.y_training)

    def set_max_trace_size(self):
        """
        Calcola la lunghezza di trace massima all'interno dell'istogramma
        :return lunghezza massima riscontrata.
        """
        max_length = 0
        for trace in self.trace_histogram:
            if len(trace) > max_length:
                max_length = len(trace)
        return max_length

    def find_variant_seeds(self, min_number_of_seeds, use_pareto):
        """
        Calcola i seed di lunghezza minima che è possibile generare dal log in quantità uguale o superione a quella
        passata come parametro; fatto ciò applica il principio di Pareto per selezionare i seed che meglio rappresentano
        le varianti presenti nell'istogramma.
        :param min_number_of_seeds: Numero minimo di seed richiesti.
        :param use_pareto: booleano che indica alla funzione se usare o meno pareto sulla lista dei seed generati.
        :return: lista di seed generati.
        """
        min_seed_lenght = 1
        candidate_seeds = Counter() if use_pareto else []
        while len(candidate_seeds) < min_number_of_seeds and min_seed_lenght < self.trace_max_lenght:
            candidate_seeds.clear() if candidate_seeds else None
            for trace in self.trace_histogram:
                if use_pareto:
                    for i in range(0, self.trace_histogram[trace]):
                        if min_seed_lenght <= len(trace):
                            candidate_seeds[tuple(x for x in (self.map_trace(trace)[0:min_seed_lenght]).tolist())] += 1
                else:
                    if min_seed_lenght <= len(trace):
                        candidate_seed = self.map_trace(trace)[0:min_seed_lenght]
                        if not any(np.array_equal(candidate_seed, x) for x in candidate_seeds):
                            candidate_seeds.append(candidate_seed.tolist())
            min_seed_lenght += 1
        if min_seed_lenght == self.trace_max_lenght and len(candidate_seeds) < min_number_of_seeds:
            raise NotEnoughSeedsException(f"Can't generate required number of seeds for this dataset. "
                                          f"Found only {len(candidate_seeds)} seeds.")
        seeds = [] if use_pareto else None
        if use_pareto:
            counter = 0
            top_variants = 0
            candidate_seeds = {k: v for k, v in sorted(candidate_seeds.items(), key=lambda x: x[1], reverse=True)}
            while counter / self.trace_window_size < 0.8:
                counter += list(candidate_seeds.values())[top_variants]
                top_variants += 1
            for i in range(0, top_variants):
                seeds.append(list(list(candidate_seeds.keys())[i]))
        return seeds if use_pareto else candidate_seeds


    def find_hyperparameters(self):

        outfile = open(self.dataset_name + '.log', 'w')
        trials = Trials()
        best = fmin(self.train_and_evaluate_model, space, algo=tpe.suggest, max_evals=20, trials=trials,
                    rstate=np.random.RandomState(42))
        self.best_params = hyperopt.space_eval(space, best)

        outfile.write("\nHyperopt trials")
        outfile.write("\ntid,loss,learning_rate,n_modules,batch_size,time,n_epochs,n_params,perf_time")
        for trial in trials.trials:
            outfile.write("\n%d,%f,%f,%d,%d,%s,%d,%d,%f" % (trial['tid'],
                                                               trial['result']['loss'],
                                                               trial['misc']['vals']['learning_rate'][
                                                                   0],
                                                               trial['misc']['vals']['num_transformer_blocks'][0],
                                                               trial['misc']['vals']['batch_size'][
                                                                   0] + 5,
                                                               (trial['refresh_time'] - trial[
                                                                   'book_time']).total_seconds(),
                                                               trial['result']['n_epochs'],
                                                               trial['result']['n_params'],
                                                               trial['result']['time']))

        outfile.write("\n\nBest parameters:")
        outfile.write("\nModel parameters: %d" % self.best_numparameters)
        outfile.write('\nBest Time taken: %f' % self.best_time)
        self.best_model.save("model/generate_" + self.dataset_name + ".h5")


    def build_neural_network_model(self):
        """
        La funzione si occupa di istanziare e addestrare una rete neurale utilizzando il dataset generato in precedenza.
        Al termine dell'addestramento il modello viene scritto su disco.
        """
        self.x_training = np.asarray(self.x_training)
        outsize = len(np.unique(self.y_training))
        self.y_training = to_categorical(self.y_training)
        unique_events = len(self.mapping_dictionary)
        X_train, X_val, Y_train, Y_val = train_test_split(self.x_training, self.y_training, test_size=0.2,
                                                          random_state=42, shuffle=True)
        size_act = (unique_events + 1) // 2
        input_act = Input(shape=(self.examples_size,), dtype='int32', name='input_act')
        x = Embedding(output_dim=size_act, input_dim=unique_events + 1, input_length=self.examples_size)(
            input_act)

        n_layers = int(self.best_params["num_transformer_blocks"])

        for i in range(n_layers):
            attention_output = layers.MultiHeadAttention(num_heads=int(self.best_params["num_heads"]), key_dim=size_act)(x, x)
            x = layers.LayerNormalization()(x + attention_output)
        
            # Feed Forward Network
            ff_output = layers.Dense(int(self.best_params["cellsize"]), activation='relu')(x)
            ff_output = layers.Dense(size_act)(ff_output)
            x = layers.LayerNormalization()(x + ff_output)
            
            # Optional Dropout to prevent overfitting
            x = layers.Dropout(self.best_params["dropout"])(x)
        x = layers.GlobalAveragePooling1D()(x)
    
        # Dense layer for classification
        x = layers.Dense(128, activation='relu')(x)

        output = Dense(outsize, activation='softmax')(x)
        model = Model(inputs=input_act, outputs=output)
        
        opt = Adam(learning_rate=self.best_params['learning_rate'])
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=20)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
        
        model.fit(X_train, Y_train, epochs=200, batch_size=self.best_params['batch_size'], verbose=0, callbacks=[early_stopping, lr_reducer],
                  validation_data=(X_val, Y_val))
        model.save("model/generate_" + self.dataset_name + ".h5")

    def get_model(self, params, outsize):
        #n_layers = int(self.best_params["num_transformer_blocks"])
        unique_events = len(self.mapping_dictionary)

        size_act = (unique_events + 1) // 2
        input_act = Input(shape=(self.examples_size,), dtype='int32', name='input_act')
        x = Embedding(output_dim=size_act, input_dim=unique_events + 1, input_length=self.examples_size)(
            input_act)

        
        n_layers = int(params["num_transformer_blocks"])

        for i in range(n_layers):
            attention_output = layers.MultiHeadAttention(num_heads=int(params["num_heads"]), key_dim=size_act)(x, x)
            x = layers.LayerNormalization()(x + attention_output)
        
            # Feed Forward Network
            ff_output = layers.Dense(int(params["cellsize"]), activation='relu')(x)
            ff_output = layers.Dense(size_act)(ff_output)
            x = layers.LayerNormalization()(x + ff_output)
            
            # Optional Dropout to prevent overfitting
            x = layers.Dropout(params["dropout"])(x)
            #x = tf.keras.layers.Dropout(self.best_params["dropout"])(x)
        x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layer for classification
        x = layers.Dense(128, activation='relu')(x)

        out_a = Dense(outsize, activation='softmax', kernel_initializer='glorot_uniform', name='output_a')(x)

        model = Model(inputs=input_act, outputs=out_a)
        opt = Adam(learning_rate=params["learning_rate"])
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
        return model


    def train_and_evaluate_model(self, params):
        self.x_training = np.asarray(self.x_training)
        outsize = len(np.unique(self.y_training))
        X_train, X_val, Y_train, Y_val = train_test_split(self.x_training, to_categorical(self.y_training), test_size=0.2,
                                                          random_state=42, shuffle=True)
        start_time = perf_counter()
        model = self.get_model(params, outsize)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)

        history = model.fit(X_train, Y_train,
                            validation_data=(X_val, Y_val), verbose=0,
                            callbacks=[early_stopping, lr_reducer],
                            batch_size=2 ** params['batch_size'], epochs=200)

        scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
        score = min(scores)
        end_time = perf_counter()

        if self.best_score > score:
            self.best_score = score
            self.best_model = model
            self.best_numparameters = model.count_params()
            self.best_time = end_time - start_time



        return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']),
                'n_params': model.count_params(), 'time': end_time - start_time}

    def predict_variants(self):
        """
        Genera una lista contenente le varianti predette dal modello partendo dai seed.
        :return: summary generato
        """
        summary = Counter()
        model = load_model("model/generate_" + self.dataset_name + ".h5")
        if model is None:
            raise ModelNotFound("No model found.")
        for seed in self.variant_seeds:
            while True:
                activity_window = np.zeros(self.examples_size)
                seed_length = len(seed)
                values = np.asarray(seed[0:]) if seed_length <= self.examples_size else \
                    np.asarray(seed[seed_length - self.examples_size:seed_length])
                activity_window[self.examples_size - values.size:] = values
                y_pred = model.predict([tuple(activity_window)])
                y_pred = y_pred.argmax(axis=1)
                y_pred = self.le.inverse_transform([y_pred])
                if y_pred == [self.mapping_dictionary.get(FINAL_ACTIVITY_SYMBOL)]:
                    break
                else:
                    seed.append(y_pred[0])
                    if len(seed) == self.trace_max_lenght:
                        break
            summary[self.rebuild_trace(seed)] += 1
        return summary

    def rebuild_trace(self, mapped_trace):
        """
        Converte array di interi nella variante corrispondente costituita da una lista di activity
        :param mapped_trace: array da convertire
        :return: lista di activity
        """
        trace = []
        for activity in mapped_trace:
            for key, value in self.mapping_dictionary.items():
                if activity == value:
                    trace.append(key)
                    break
        return tuple(trace)

    def build_summary(self, updated_variants_histogram):
        """
        Costruisce un riassunto dei log iniziali partendo dalla ricerca dei seed per poi estenderli con le activity
        predette dalla rete.
        :param updated_variants_histogram istogramma trace-frequenza da utilizzare per generare il riassunto
        """
        self.trace_histogram = updated_variants_histogram
        self.trace_max_lenght = self.set_max_trace_size()
        self.variant_seeds = self.find_variant_seeds(1, False)
        return self.predict_variants()

    def is_already_trained(self):
        """
        Verifica se esiste già un modello generato partendo dal dataset dato.
        """
        return Path("model/generate_" + self.dataset_name + ".h5").is_file()


class NotEnoughSeedsException(Exception):
    """"
    Sollevata quando non è possibile generare dall'istogramma di partenza il numero richiesto di seed.
    """
    pass


class ModelNotFound(Exception):
    """"
    Sollevata quando si tenta di utilizzare il modello prima che venga istanziato.
    """
    pass
