import torch, pandas as pd, numpy as np, os
# from bindsnet.network import Network
# from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
# from bindsnet.network.topology import Connection
# from bindsnet.network.monitors import Monitor
# from bindsnet.analysis.plotting import plot_spikes, plot_voltages
# from bindsnet.learning import PostPre
# import torch.nn.functional as F

import argparse
import json
import optuna

import numpy as np

from utils import *
date_starting_trials = datetime.now().strftime('%Y_%m_%d-%H_%M')  # Format includes year, month, day, hour and minute

# # Add this near the top of the file after imports
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def experiment(nu1, nu2, a, r, n, threshold, decay, T, expansion, path, n_trial, trial=None):

    #Lectura de datos:
    #Esperamos que estos datos tengan las columnas 'label' y 'value'.

    data=pd.read_csv(path,na_values=['NA'])

    #Asegurarse de que los tipos sean correctos:
    data['value']=data['value'].astype('float64')
    data['label']=data['label'].astype('Int64')

    #Y ponemos a 0 los valores nulos del label para no tener problemas al filtrar por esta columna:
    data.loc[data['label'].isna(),'label']=0

    split = len(data) // 2

    data_train=data[:split]
    data_test=data[split:]

    #Reseteamos el índice:
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    #Expandimos:
    data_train['label']=expandir(data_train['label'],expansion)

    #Sacamos máximos y mínimos:
    minimo=min(data_train['value'][data_train['label']!=1])
    maximo=max(data_train['value'][data_train['label']!=1])

    #Declaramos el vector de cuantiles. Para ello, tomamos el máximo y mínimo de los datos de entrenamiento (esto hay que sacarlo de esos datos)

    amplitud=maximo-minimo
    cuantiles=torch.FloatTensor(np.arange(minimo-a*amplitud,maximo+amplitud*a,(maximo-minimo)*r))

    #Ahora, establecemos el valor de snn_input_layer_neurons_size, que será el número de neuronas de la capa de entrada:
    snn_input_layer_neurons_size=len(cuantiles)-1

    #Crea la red.
    network, source_monitor, target_monitor, conv_monitor = crear_red(snn_input_layer_neurons_size,decay,threshold,nu1,nu2,n,T,use_conv_layer=use_conv_layer,device=device)

    #Dividimos el train en secuencias:
    data_train=dividir(data_train,T)

    #Paddeamos el test:
    data_test=padd(data_test,T)

    #En este punto, entrenamos para cada secuencia consecutiva del train:

    #Para cada secuencia del train, tenemos que pasarla y entrenar la red:
    network.learning=True

    for s in data_train:
        secuencias2train=convertir_data(s,T,cuantiles,snn_input_layer_neurons_size,is_train=True,device=device)
        print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
        spikes_input,spikes,spikes_conv,network=ejecutar_red(secuencias2train,network,source_monitor,target_monitor,conv_monitor,T,use_conv_layer=use_conv_layer,device=device)
        #Reseteamos los voltajes:
        network=reset_voltajes(network)

    #Ahora, el test:
    network.learning=False
    secuencias2test=convertir_data(data_test,T,cuantiles,snn_input_layer_neurons_size,is_train=False,device=device)

    print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
    spikes_input,spikes,spikes_conv,network=ejecutar_red(secuencias2test,network,source_monitor,target_monitor,conv_monitor,T,use_conv_layer=use_conv_layer,device=device)

    mse_B, mse_C = guardar_resultados(spikes,spikes_conv,data_test,n,snn_input_layer_neurons_size,n_trial,date_starting_trials,dataset_name,snn_process_layer_neurons_size,trial=trial)
    return mse_B, mse_C

def objective(trial):

    print(f"Running trial: {trial.number + 1}")
    config = {
        'nu1': trial.suggest_float('nu1', -0.5, 0.5),
        'nu2': trial.suggest_float('nu2', -0.5, 0.5),
        'threshold': trial.suggest_float('threshold', -65, -50),
        'decay': trial.suggest_float('decay', 80, 150),
    }
    print(f"config: {config}")

    #Establecemos valores para los parámetros que nos interesan:
    # nu1_pre=0.1 #Actualización de pesos presinápticos en la capa A. Valores positivos penalizan y negativos excitan.
    # nu1_post=-0.1 #Actualización de pesos postsinápticos en la capa A. Valores postivos excitan y negativos penalizan.

    # nu2_pre=0.1 #Actualización de pesos presinápticos en la capa B. Valores positivos penalizan y negativos excitan.
    # nu2_post=-0.1 #Actualización de pesos postsinápticos en la capa B. Valores postivos excitan y negativos penalizan.

    #Parámetros que definen la amplitud del rango de cuantiles.
    #La idea es que el valor mínimo para la codificación sea inferior al mínimo de los datos de entrenamiento, por un margen. El valor máximo debe ser también  mayor que el máximo de los datos por un margen.
    #Para ello, nos inventamos la variable a, que será la proporción del rango de datos de entrenamiento que inflamos por encima y por debajo:
    a=0.1
    #La resolución, r, indica cuán pequeños tomamos los rangos al codificar:
    r=0.05

    #Número de neuronas en la capa B.

    #Umbral de disparo de las neuronas LIF:
    # threshold=-52

    # #Decaimiento, en tiempo, de las neuronas LIF:
    # decay=100

    T = 250 #Tiempo de exposición. Puede influir por la parte del entrenamiento, en la inferencia no porque los voltajes se conservan.
    #Usar el máximo de T para evitar problemas con los periodos de datos.
    expansion=100
    
    nu1=(config['nu1'],config['nu1'])
    nu2=(config['nu2'],config['nu2'])
    try:
        # Run the experiment with GPU enabled by default
        mse_B, mse_C = experiment(nu1, nu2, a, r, snn_process_layer_neurons_size, config['threshold'], config['decay'], T, expansion, path, trial.number + 1,trial=trial)
        
        return mse_B
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return a very low score for failed trials
        return float('inf')

if __name__ == "__main__":
    start_time = datetime.now()  # Add this line to track start time
    
    parser = argparse.ArgumentParser(description='Optimización de hiperparámetros con Optuna.')
    path='Nuevos datasets\\iops\\preliminar\\train_procesado_javi\\1c35dbf57f55f5e4_filled.csv'
    dataset_name=path.split('\\')[1]
    snn_process_layer_neurons_size=100
    use_conv_layer=False
    # path='Nuevos datasets/Callt2/preliminar\\train_label_filled.csv'
    parser.add_argument('-d', '--data_path', type=str, default='Nuevos datasets\\Callt2\\preliminar\\train_label_filled.csv', help='Ruta al archivo de datos CSV')
    parser.add_argument('-n', '--n_trials', type=int, default=100, help='Número de trials para Optuna')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu', help='Device to use (cpu/gpu)')
    args = parser.parse_args()


    device = torch.device("cuda" if (args.device == "gpu") and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    end_time = datetime.now()  # Add this line to track end time
    duration = end_time - start_time  # Calculate duration

    print('Mejor configuración encontrada:')
    print(study.best_params)
    print(f'Mejor MSE_B: {study.best_value}')
    print(f'Duración total: {duration}')

    # Guardar la mejor configuración
    base_path = f'resultados/{dataset_name}/n_{snn_process_layer_neurons_size}/{date_starting_trials}'
    os.makedirs(base_path, exist_ok=True)
    best_trial_number = study.best_trial.number
    results = {
        "best_params": study.best_params,
        "best_trial": best_trial_number+1,
        "snn_process_layer_neurons_size": snn_process_layer_neurons_size,
        "device": str(device),  # Convert device to string
        "use_conv_layer": use_conv_layer,
        "best_mse_B": study.best_value,
        "amoumt_of_trials": args.n_trials,
        "duration_seconds": duration.total_seconds(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat()
    }
    with open(f"{base_path}/best_config.json", "w") as f:
        json.dump(results, f, indent=4)
