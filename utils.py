from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
import torch, pandas as pd, numpy as np, os
from bindsnet.learning import PostPre
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os
import json


def reset_voltajes(network, device='cpu'):
    network.layers['B'].v = torch.full(network.layers['B'].v.shape, -65, device=device)
    return network


def dividir(data,minimo):
    #Función que divide los datos de entrenamiento, para considerar aisladamente cada subsecuencia de datos normales.
    #Tomamos los intervalos:
    intervals = []
    in_sequence = False
    
    #Iteramos para identificar los intervalos:
    for i in range(len(data)):
        if data.loc[i, 'label'] == 0:
            if not in_sequence:
                start_idx = i
                in_sequence = True
            end_idx = i+1
        else:
            if in_sequence:
                intervals.append((start_idx, end_idx))
                in_sequence = False
    
    # Agrega la posición del último elemento de los datos de entrada:
    if in_sequence:
        intervals.append((start_idx, end_idx))
    
    #Creamos un dataframe con los intervalos encontrados:
    intervals_df = pd.DataFrame(intervals, columns=['inicio', 'final'])
    
    subs=[]
    #Iteramos para dividir:
    for i,row in intervals_df.iterrows():
        inicio_tmp=row['inicio']
        final_tmp=row['final']
        if final_tmp-inicio_tmp>=minimo:
            subs.append(data.iloc[inicio_tmp:final_tmp].reset_index(drop=True))
    
    return subs


def padd(data, T):
    lon = len(data)
    # Calcular el múltiplo más cercano de T superior al número actual de filas
    lon2 = ((lon // T) + 1) * T
    # Calcular el número de filas adicionales necesarias
    lon_adicional = lon2 - lon
    
    # Crear un DataFrame con filas adicionales llenas de NaN
    if lon_adicional > 0:
        datanul = pd.DataFrame(np.nan, index=range(lon_adicional), columns=data.columns)
        # Concatenar el DataFrame original con el DataFrame de padding
        data = pd.concat([data, datanul], ignore_index=True)
    
    return data


def expandir(serie, n):
    # Crea gemelo de la serie:
    serie2 = np.zeros_like(serie)
    
    # Identificar los índices donde hay un 1:
    indices = np.where(serie == 1)[0]
    
    # Poner a 1 los valores en el rango [índice-n, índice+n]
    for idx in indices:
        start = max(0, idx - n)
        end = min(len(serie), idx + n + 1)
        serie2[start:end] = 1
    
    return pd.Series(serie2, index=serie.index)


#Función para convertir a spikes las entradas:
def podar(x,q1,q2,cuantiles=None):
    #Función que devuelve 1 (spike) si x está en el rango [q1,q2), y 0 en caso contrario.
    #Es parte de la codificación de los datos.
    
    s=torch.zeros_like(x)
    
    s[(x>=q1) & (x<q2)]=1
    return s


def convertir_data(data, T, cuantiles, snn_input_layer_neurons_size, is_train=False, device='cpu'):
    # Move cuantiles to GPU
    print('convertir_data')
    print(device)
    
    cuantiles = cuantiles.to(device)
    
    # Convert series to GPU tensor
    serie = torch.FloatTensor(data['value']).to(device)
    
    #Tomamos la longitud de la serie.
    long=serie.shape[0]
    
    #Los valores inferiores al mínimo del vector de cuantiles se sustituyen por ese mínimo.
    serie[serie<torch.min(cuantiles)]=torch.min(cuantiles)
    serie[serie>torch.max(cuantiles)]=torch.max(cuantiles)
    
    #Construimos el tensor con los datos codificados.
    serie2input=torch.cat([serie.unsqueeze(0)] * snn_input_layer_neurons_size, dim=0)
    
    for i in range(snn_input_layer_neurons_size):
        serie2input[i,:]=podar(serie2input[i,:],cuantiles[i],cuantiles[i+1])
    
    #Lo dividimos en función del tiempo de exposición T:
    secuencias = torch.split(serie2input,T,dim=1)
    
    if is_train:
        secuencias=secuencias[0:len(secuencias)-1]
    
    return secuencias


# Function to create a Gaussian kernel
def create_gaussian_kernel(kernel_size=5, sigma=1.0, device='cpu'):
    # Create a 1D Gaussian kernel directly on GPU
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
    gaussian = torch.exp(-x**2 / (2*sigma**2))
    gaussian = gaussian / gaussian.sum()  # Normalize
    return gaussian.view(1, 1, -1)  # Shape for 1D convolution


def crear_red(snn_input_layer_neurons_size, decaimiento, umbral, nu1, nu2, n, T, use_conv_layer=True, device='cpu'):
    # Create the network
    print('crear_red')
    print(device)
    
    network = Network(dt=1.0, learning=True).to(device)
    
    # Create layers and move to GPU
    source_layer = Input(n=snn_input_layer_neurons_size, traces=True).to(device)
    target_layer = LIFNodes(n=n, traces=True, thresh=umbral, tc_decay=decaimiento).to(device)
    
    network.add_layer(layer=source_layer, name="A")
    network.add_layer(layer=target_layer, name="B")
    
    conv_layer = None
    if use_conv_layer:
        conv_layer = LIFNodes(n=n, traces=True, thresh=umbral, tc_decay=decaimiento).to(device)
        network.add_layer(layer=conv_layer, name="C")
    
    # Create forward and recurrent connections...
    forward_connection = Connection(
        source=source_layer,
        target=target_layer,
        w=(0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n)).to(device),
        update_rule=PostPre, 
        nu=nu1
    ).to(device)
    
    network.add_connection(connection=forward_connection, source="A", target="B")
    
    recurrent_connection = Connection(
        source=target_layer,
        target=target_layer,
        w=(0.025 * (torch.eye(target_layer.n) - 1)).to(device), 
        update_rule=PostPre, 
        nu=nu2
    ).to(device)
    
    network.add_connection(connection=recurrent_connection, source="B", target="B")
    
    # Add convolutional layer and connection only if use_conv_layer is True
    if use_conv_layer:
        kernel = create_gaussian_kernel(kernel_size=5, sigma=1.0, device=device).repeat(n, 1, 1)
        kernel_size = 5
        
        weights = torch.zeros(n, n, device=device)
        center = kernel_size // 2
        for i in range(n):
            start = max(0, i - center)
            end = min(n, i + center + 1)
            k_start = max(0, center - i)
            k_end = kernel_size - max(0, i + center + 1 - n)
            weights[i, start:end] = kernel[0,0,k_start:k_end]
        
        conv_connection = Connection(
            source=target_layer,
            target=conv_layer,
            w=weights,
            update_rule=PostPre,
            nu=nu2,
            norm=0.5 * kernel_size
        ).to(device)
        
        network.add_connection(conv_connection, "B", "C")
    
    #Creamos los monitores. Sirven para registrar los spikes y voltajes:
    #Spikes de entrada (para depurar que se esté haciendo bien, si se quiere):
    # Create monitors
    source_monitor = Monitor(
        obj=source_layer,
        state_vars=("s",),  #Registramos sólo los spikes.
        time=T,
    )
    #Spikes de la capa recurrente (lo que nos interesa):
    target_monitor = Monitor(
        obj=target_layer,
        state_vars=("s", "v"),  #Registramos spikes y voltajes, por si nos interesa lo segundo también.
        time=T,
    )
    
    network.add_monitor(monitor=source_monitor, name="X")
    network.add_monitor(monitor=target_monitor, name="Y")
    
    conv_monitor = None
    if use_conv_layer:
        conv_monitor = Monitor(
            obj=conv_layer,
            state_vars=("s", "v"),
            time=T,
        )
        network.add_monitor(monitor=conv_monitor, name="Conv_mon")
    
    return [network, source_monitor, target_monitor, conv_monitor]


def ejecutar_red(secuencias, network, source_monitor, target_monitor, conv_monitor, T, use_conv_layer=True, device='cpu'):
    sp0, sp1, sp_conv = [], [], []
    
    print('ejecutar_red')
    print(device)
    j = 1
    for i in secuencias:
        print(f'Ejecutando secuencia {j}')
        j += 1
        
        inputs = {'A': i.T.to(device)}
        network.run(inputs=inputs, time=T)
        
        spikes = {
            "X": source_monitor.get("s").to(device),
            "B": target_monitor.get("s").to(device)
        }
        
        if use_conv_layer and conv_monitor is not None:
            spikes["C"] = conv_monitor.get("s").to(device)
        
        b_spikes = spikes["B"].float()
        b_spikes_sum = b_spikes.sum(dim=2).transpose(0, 1)
        
        sp0.append(spikes['X'].cpu().sum(axis=2))
        sp1.append(spikes['B'].cpu().sum(axis=2))
        
        if use_conv_layer:
            kernel = create_gaussian_kernel(device=device)
            conv_spikes = F.conv1d(b_spikes_sum, kernel, padding='same')
            sp_conv.append(conv_spikes.cpu().squeeze())
        
        network = reset_voltajes(network, device=device)
    
    sp0 = torch.cat(sp0).cpu().detach().numpy()
    sp1 = torch.cat(sp1).cpu().detach().numpy()
    
    if use_conv_layer:
        sp_conv = torch.cat(sp_conv).cpu().detach().numpy()
    else:
        sp_conv = None
    
    return [sp0, sp1, sp_conv, network]


def guardar_resultados(spikes, spikes_conv, data_test, n, snn_input_layer_neurons_size, n_trial, date_starting_trials, dataset_name, snn_process_layer_neurons_size, trial):
    # Create directory structure
    base_path = f'resultados/{dataset_name}/n_{snn_process_layer_neurons_size}/{date_starting_trials}/trial_{n_trial}'
    os.makedirs(base_path, exist_ok=True)

    # Save spikes
    np.savetxt(f'{base_path}/spikes', spikes, delimiter=',')
    
    # Only save spikes_conv if it exists
    if spikes_conv is not None:
        np.savetxt(f'{base_path}/spikes_conv', spikes_conv, delimiter=',')

    # Convert and save labels - handle NA values properly
    labels = data_test['label'].replace([np.inf, -np.inf], np.nan)
    labels = labels.astype(float)
    labels = labels.to_numpy()
    np.savetxt(f'{base_path}/label', labels, delimiter=',')

    # Convert and save values - handle NA values properly 
    values = data_test['value'].replace([np.inf, -np.inf], np.nan)
    values = values.astype(float)
    values = values.to_numpy()
    np.savetxt(f'{base_path}/value', values, delimiter=',')

    # Save timestamps
    timestamps = data_test['timestamp'].replace([np.inf, -np.inf], np.nan)
    timestamps = timestamps.astype(float)
    timestamps = timestamps.to_numpy()
    np.savetxt(f'{base_path}/timestamp', timestamps, delimiter=',')

    # Create DataFrame with 1D arrays
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'label': labels
    })

    # Save to CSV with same format as original
    results_df.to_csv(f'{base_path}/data_test.csv', 
                     index=False,
                     float_format='%.6f')

    # Reshape/flatten spikes to 1D if needed
    spikes_1d = spikes.sum(axis=1) if len(spikes.shape) > 1 else spikes

    # Create DataFrame with 1D arrays
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'label': spikes_1d
    })

    # Save to CSV with same format as original
    results_df.to_csv(f'{base_path}/results.csv', 
                     index=False,
                     float_format='%.6f')

    # Only process convolutional layer results if spikes_conv exists
    mse_C = None
    if spikes_conv is not None:
        spikes_conv_1d = spikes_conv.sum(axis=1) if len(spikes_conv.shape) > 1 else spikes_conv

        results_conv_df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'label': spikes_conv_1d
        })

        results_conv_df.to_csv(f'{base_path}/results_conv.csv', 
                             index=False,
                             float_format='%.6f')

        # Calculate MSE for conv layer
        spikes_conv_1d = spikes_conv_1d.astype(float)
        spikes_conv_1d = np.nan_to_num(spikes_conv_1d, nan=0.0)
        mse_C = mean_squared_error(y_true, spikes_conv_1d)    
        print("MSE capa C:", mse_C)
        with open(f'{base_path}/MSE_capa_C', 'w') as n2:
            n2.write(f'{mse_C}\n')

    # with open(f'{base_path}/n1', 'w') as n1:
    #     n1.write(f'{snn_input_layer_neurons_size}\n')

    # with open(f'{base_path}/n2', 'w') as n2:
    #     n2.write(f'{n}\n')

    # Calculate MSE for layer B
    y_true = data_test['label'].astype(float).to_numpy()
    y_true = np.nan_to_num(y_true, nan=0.0)

    spikes_1d = spikes_1d.astype(float)
    spikes_1d = np.nan_to_num(spikes_1d, nan=0.0)

    mse_B = mean_squared_error(y_true, spikes_1d)
    # print("MSE capa B:", mse_B)
    # with open(f'{base_path}/MSE_capa_B', 'w') as n2:
    #     n2.write(f'{mse_B}\n')
        
    info = {
        "nu1": trial.params['nu1'],
        "nu2": trial.params['nu2'],
        "threshold": trial.params['threshold'],
        "decay": trial.params['decay'],
        "mse_B": mse_B,
        "mse_C": mse_C,
        "ssn_input_layer_neurons_size": snn_input_layer_neurons_size,
        "snn_process_layer_neurons_size": snn_process_layer_neurons_size,    
    }
    with open(f"{base_path}/best_config.json", "w") as f:
        json.dump(info, f, indent=4)
        
    return mse_B, mse_C