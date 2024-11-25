from collections import deque

import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
import tensorflow as tf

class MyPredictor(object):
    """An example Predictor for an AI Platform custom prediction routine."""

    def __init__(self, clasificador, scallerAll, scallerc0, scallerc1, scallerc2, scallerc3, scallerc4, scallerc5,
                   loaded_graph0, loaded_graph1, loaded_graph2, loaded_graph3, loaded_graph4, loaded_graph5,
                   sess0, sess1, sess2, sess3, sess4, sess5):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._clasificador = clasificador
        self._scalerAll = scallerAll
        self._scallerc0 = scallerc0
        self._scallerc1 = scallerc1
        self._scallerc2 = scallerc2
        self._scallerc3 = scallerc3
        self._scallerc4 = scallerc4
        self._scallerc5 = scallerc5
        self._loaded_graph0 = loaded_graph0
        self._loaded_graph1 = loaded_graph1
        self._loaded_graph2 = loaded_graph2
        self._loaded_graph3 = loaded_graph3
        self._loaded_graph4 = loaded_graph4
        self._loaded_graph5 = loaded_graph5
        self._sess0 = sess0
        self._sess1 = sess1
        self._sess2 = sess2
        self._sess3 = sess3
        self._sess4 = sess4
        self._sess5 = sess5
        
        #Create Sliding Windows
        self.len_ventana_rampas = 8
        self.ventanaRampas = self.slidingWindows(self.len_ventana_rampas)
        self.len_ventana_anomalia = 4
        self.ventanaAnomalia = self.slidingWindows(self.len_ventana_anomalia)

    def predict(self, instances, **kwargs):
        """Performs custom prediction.

        Preprocesses inputs, then performs prediction using the trained Keras
        model.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results.
        """
        array31 = np.asarray(instances)
        array34 = self.addItemSlidingWindows(array31)
        array31AE, cluster = self.clasificacion(array34)
        coste = self.call_ae(array31AE, cluster)
        error = self.applyThreshold(cluster, coste)
        return error, cluster

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of MyPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the trained Keras
                model and the pickled preprocessor instance. These are copied
                from the Cloud Storage model directory you provide when you
                deploy a version resource.

        Returns:
            An instance of `MyPredictor`.
        """

        clasificador_path = os.path.join(model_dir, 'Clasificador/GMMClustering.save')
        clasificador = joblib.load(clasificador_path)
        scallerAll_path = os.path.join(model_dir, 'Scalers/scalerAll.save')
        scallerAll = joblib.load(scallerAll_path)
        scallerc0_path = os.path.join(model_dir, 'Scalers/scalerC0.save')
        scallerc0 = joblib.load(scallerc0_path)
        scallerc1_path = os.path.join(model_dir, 'Scalers/scalerC1.save')
        scallerc1 = joblib.load(scallerc1_path)
        scallerc2_path = os.path.join(model_dir, 'Scalers/scalerC2.save')
        scallerc2 = joblib.load(scallerc2_path)
        scallerc3_path = os.path.join(model_dir, 'Scalers/scalerC3.save')
        scallerc3 = joblib.load(scallerc3_path)
        scallerc4_path = os.path.join(model_dir, 'Scalers/scalerC4.save')
        scallerc4 = joblib.load(scallerc4_path)
        scallerc5_path = os.path.join(model_dir, 'Scalers/scalerC5.save')
        scallerc5 = joblib.load(scallerc5_path)

        model0_path = os.path.join(model_dir, 'AE/cluster0/savedmodel')
        model1_path = os.path.join(model_dir, 'AE/cluster1/savedmodel')
        model2_path = os.path.join(model_dir, 'AE/cluster2/savedmodel')
        model3_path = os.path.join(model_dir, 'AE/cluster3/savedmodel')
        model4_path = os.path.join(model_dir, 'AE/cluster4/savedmodel')
        model5_path = os.path.join(model_dir, 'AE/cluster5/savedmodel')

        loaded_graph0 = tf.Graph()
        loaded_graph1 = tf.Graph()
        loaded_graph2 = tf.Graph()
        loaded_graph3 = tf.Graph()
        loaded_graph4 = tf.Graph()
        loaded_graph5 = tf.Graph()

        sess0 = tf.Session(graph=loaded_graph0)
        sess1 = tf.Session(graph=loaded_graph1)
        sess2 = tf.Session(graph=loaded_graph2)
        sess3 = tf.Session(graph=loaded_graph3)
        sess4 = tf.Session(graph=loaded_graph4)
        sess5 = tf.Session(graph=loaded_graph5)

        tf.saved_model.loader.load(sess0, [tf.saved_model.tag_constants.SERVING], model0_path)
        tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], model1_path)
        tf.saved_model.loader.load(sess2, [tf.saved_model.tag_constants.SERVING], model2_path)
        tf.saved_model.loader.load(sess3, [tf.saved_model.tag_constants.SERVING], model3_path)
        tf.saved_model.loader.load(sess4, [tf.saved_model.tag_constants.SERVING], model4_path)
        tf.saved_model.loader.load(sess5, [tf.saved_model.tag_constants.SERVING], model5_path)

        return cls(clasificador, scallerAll, scallerc0, scallerc1, scallerc2, scallerc3, scallerc4, scallerc5,
                   loaded_graph0, loaded_graph1, loaded_graph2, loaded_graph3, loaded_graph4, loaded_graph5,
                   sess0, sess1, sess2, sess3, sess4, sess5)

    #Method to create a slidingWindows
    def slidingWindows(self,n):
        windows = deque(maxlen=n)
        return windows

    #Method to add a item in a slidingWindows
    def addItemSlidingWindows(self, tupla):
        mean = 0.0
        threshold = 0.7
        #Windows is not full?
        if(len(self.ventanaRampas) < self.len_ventana_rampas):
            tupla = np.append(tupla, [False, False, True])
        #When windows is full...
        else:
            #Get mean of ActiveLoad
            arrayVentana = np.array(self.ventanaRampas)
            acumulative = 0
            for i in arrayVentana:
                acumulative = acumulative + i[0]
            mean = acumulative/self.len_ventana_rampas
            #Increase
            if((tupla[0] - threshold) > mean):
                tupla = np.append(tupla, [False, True, False])
            #Decrease
            elif((tupla[0] + threshold) < mean):
                tupla = np.append(tupla, [True, False, False])
            #Stable
            else:
                tupla = np.append(tupla, [False, False, True]) 

        #Add item in windows
        self.ventanaRampas.append(tupla)

        return tupla  

    def classifier(self, tupla, clasificador, scaler):
        tupla = tupla.reshape(1, -1)
        # Normalize
        tupla_normalizada = scaler.transform(tupla)
        dfTupla = pd.DataFrame(tupla_normalizada)
        # clasificamos la tupla
        cluster = clasificador.predict(dfTupla)
        return cluster[0]

    def normalizeByCluster(self, tupla, scaler):
        tupla = tupla.reshape(1, -1)
        tupla_normalizada = scaler.transform(tupla)
        return tupla_normalizada

    def clasificacion(self, tupla):
        # call clustering classifier
        cluster = self.classifier(tupla, self._clasificador, self._scalerAll)
        # after clasification, delete useless column
        tupla = np.delete(tupla, [31, 32, 33])
        # prepare data normalizeByCluster for the AE
        # We must change number of cluster because when training algorithms, this doesnt return
        # number of cluster ordered, we order for human understanding
        if (cluster == 3):
            tuplaAE = self.normalizeByCluster(tupla, self._scalerc0)
            cluster = 0
        elif (cluster == 0):
            tuplaAE = self.normalizeByCluster(tupla, self._scalerc1)
            cluster = 1
        elif (cluster == 4):
            tuplaAE = self.normalizeByCluster(tupla, self._scalerc2)
            cluster = 2
        elif (cluster == 1):
            tuplaAE = self.normalizeByCluster(tupla, self._scalerc3)
            cluster = 3
        elif (cluster == 2):
            tuplaAE = self.normalizeByCluster(tupla, self._scalerc4)
            cluster = 4
        elif (cluster == 5):
            tuplaAE = self.normalizeByCluster(tupla, self._scalerc5)
            cluster = 5

        return tuplaAE, cluster

    def call_ae(self, tupla, cluster):
        if cluster == 0:
            input_data = self._loaded_graph0.get_tensor_by_name('X:0')
            coste = self._loaded_graph0.get_tensor_by_name('COST:0')
            cost = self.sess0.run(coste, feed_dict={input_data: tupla})
            return cost
        elif cluster == 1:
            input_data = self._loaded_graph1.get_tensor_by_name('X:0')
            coste = self._loaded_graph1.get_tensor_by_name('COST:0')
            cost = self.sess1.run(coste, feed_dict={input_data: tupla})
            return cost
        elif cluster == 2:
            input_data = self._loaded_graph2.get_tensor_by_name('X:0')
            coste = self._loaded_graph2.get_tensor_by_name('COST:0')
            cost = self.sess2.run(coste, feed_dict={input_data: tupla})
            return cost
        elif cluster == 3:
            input_data = self._loaded_graph3.get_tensor_by_name('X:0')
            coste = self._loaded_graph3.get_tensor_by_name('COST:0')
            cost = self.sess3.run(coste, feed_dict={input_data: tupla})
            return cost
        elif cluster == 4:
            input_data = self._loaded_graph4.get_tensor_by_name('X:0')
            coste = self._loaded_graph4.get_tensor_by_name('COST:0')
            cost = self.sess4.run(coste, feed_dict={input_data: tupla})
            return cost
        elif cluster == 5:
            input_data = self._loaded_graph5.get_tensor_by_name('X:0')
            coste = self._loaded_graph5.get_tensor_by_name('COST:0')
            cost = self.sess5.run(coste, feed_dict={input_data: tupla})
            return cost

    def applyThreshold(self, cluster, coste):
        anomaliaTupla = False
        if (cluster == 0):
            if (coste > 0.001):
                anomaliaTupla = True
        elif (cluster == 1):
            if (coste > 0.012):
                anomaliaTupla = True
        elif (cluster == 2):
            if (coste > 0.008):
                anomaliaTupla = True
        elif (cluster == 3):
            if (coste > 0.055):
                anomaliaTupla = True
        elif (cluster == 4):
            if (coste > 0.01):
                anomaliaTupla = True
        elif (cluster == 5):
            if (coste > 0.005):
                anomaliaTupla = True

        # Add anomalia from cluster to windows
        hayAnomalia = self.addAnomaliaSlidingWindows(self.ventanaAnomalia, anomaliaTupla)

        return hayAnomalia


    def addAnomaliaSlidingWindows(self, ventana, anomalia):
        anomaliaVentana = False
        ventana.append(anomalia)
        # Si la ventana esta llena comprobamos el numero de anomalias en la ventana
        if (len(ventana) == self.len_ventana_anomalia):
            nAnomalias = ventana.count(True)
            # Si la ventana solo tiene anomalias
            if (nAnomalias == self.len_ventana_anomalia):
                anomaliaVentana = True

        return anomaliaVentana


