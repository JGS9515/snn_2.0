import torch, pandas as pd, numpy as np, os
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre

#Código ppal para lanzar la experimentación para la detección de anomalías con bindsnet y STDP.
#Este código se usaría como base para iterar sobre las distintas combinaciones de parámetros.

#Establecemos valores para los parámetros que nos interesan:
nu1_pre=0.1 #Actualización de pesos presinápticos en la capa A. Valores positivos penalizan y negativos excitan.
nu1_post=-0.1 #Actualización de pesos postsinápticos en la capa A. Valores postivos excitan y negativos penalizan.

nu2_pre=0.1 #Actualización de pesos presinápticos en la capa B. Valores positivos penalizan y negativos excitan.
nu2_post=-0.1 #Actualización de pesos postsinápticos en la capa B. Valores postivos excitan y negativos penalizan.

#Parámetros que definen la amplitud del rango de cuantiles.
#La idea es que el valor mínimo para la codificación sea inferior al mínimo de los datos de entrenamiento, por un margen. El valor máximo debe ser también  mayor que el máximo de los datos por un margen.
#Para ello, nos inventamos la variable a, que será la proporción del rango de datos de entrenamiento que inflamos por encima y por debajo:
a=0.1
#La resolución, r, indica cuán pequeños tomamos los rangos al codificar:
r=0.01

#Número de neuronas en la capa B.
n=1000

#Umbral de disparo de las neuronas LIF:
umbral=-52

#Decaimiento, en tiempo, de las neuronas LIF:
decaimiento=100

T = 1000 #Tiempo de exposición. Puede influir por la parte del entrenamiento, en la inferencia no porque los voltajes se conservan.

#Ruta de los datos que se emplearán.
folder_data='' 


#Construimos las tuplas n1 y n2 para pasar al modelo:
nu1=(nu1_pre,nu1_post)
nu2=(nu2_pre,nu2_post)

#Declaramos el vector de cuantiles. Para ello, tomamos el máximo y mínimo de los datos de entrenamiento (esto hay que sacarlo de esos datos, claro):
amplitud=maximo-minimo
cuantiles=torch.FloatTensor(np.arange(minimo-a*amplitud,maximo+amplitud*a,(maximo-minimo)*r))

#Ahora, establecemos el valor de R, que será el número de neuronas de la capa de entrada:
R=len(cuantiles)-1


#Función para convertir a spikes las entradas:
def podar(x,q1,q2,cuantiles=None):
    #Función que devuelve 1 (spike) si x está en el rango [q1,q2), y 0 en caso contrario.
    #Es parte de la codificación de los datos.
    
    s=torch.zeros_like(x)
    
    s[(x>=q1) & (x<q2)]=1
    return s


def leer_data(input_folder,T):
    #Función que lee los datos y los prepara:
    data=pd.read_csv(input_folder)
    
    #Esta parte debe ser modificada para obtener la serie temporal de interés
    #almacenada en la variable serie.
    #
    serie=torch.FloatTensor(data[variable])
    
    #Tomamos la longitud de la serie.
    long=serie.shape[0]
    
    #Los valores inferiores al mínimo del vector de cuantiles se sustituyen por ese mínimo.
    #Los valores mayores que el máximo de los cuantiles se sustituyen por ese máximo.
    #Esto sirve para no perder spikes cuando se procesan los datos de prueba.
    
    serie[serie<torch.min(cuantiles)]=torch.min(cuantiles)
    serie[serie>torch.max(cuantiles)]=torch.max(cuantiles)
    
    #Construimos el tensor con los datos codificados. Básicamente, para cada dato de entrada tendremos una secuencia donde 
    #todos los valores son 0 menos 1, correspondiente al cuantil correspondiente al dato de entrada.
    serie2input=torch.cat([serie.unsqueeze(0)] * R, dim=0)
    
    for i in range(R):
        serie2input[i,:]=podar(serie2input[i,:],cuantiles[i],cuantiles[i+1])
    
    #Lo dividimos en función del tiempo de exposición T:
    secuencias = torch.split(serie2input,T,dim=1)
    secuencias=secuencias[0:len(secuencias)-1]
    
    return secuencias


def crear_red():
    #Aquí creamos la red.
    
    network = Network()
    
    #Creamos las capas de entrada e interna:
    source_layer = Input(n=R,traces=True)
    target_layer = LIFNodes(n=n,traces=True,thresh=umbral, tc_decay=decaimiento)
    
    network.add_layer(
        layer=source_layer, name="A"
    )
    network.add_layer(
        layer=target_layer, name="B"
    )
    
    #Creamos conexiones entre las capas de entrada y la recurrente:
    forward_connection = Connection(
        source=source_layer,
        target=target_layer,
        w=0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n),
        update_rule=PostPre, nu=nu1#nu=(1e-4, 1e-2)    # Normal(0.05, 0.01) weights.
    )
    
    network.add_connection(
        connection=forward_connection, source="A", target="B"
    )
    
    # Creamos la conexión recurrente con pesos ligeramente negativos (si no, estamos metiendo posible ruido en el procesamiento de la red):
    recurrent_connection = Connection(
        source=target_layer,
        target=target_layer,
        w=0.025 * (torch.eye(target_layer.n) - 1), 
        update_rule=PostPre, nu=nu2#nu=(1e-4, 1e-2)
    )
    
    network.add_connection(
        connection=recurrent_connection, source="B", target="B"
    )
    
    #Creamos los monitores. Sirven para registrar los spikes y voltajes:
    #Spikes de entrada (para depurar que se esté haciendo bien, si se quiere):
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
    
    
    return [network,source_monitor,target_monitor]


def ejecutar_red(secuencias,network,source_monitor,target_monitor,T):
    #Función para ejecutar la red con los datos que se quieran, ya sea para entrenamiento o evaluación.
    
    #Creamos los objetos lista en que almacenaremos los resultados:
    sp0=[]
    sp1=[]
    
    #Entrenamos:
    j=1
    for i in secuencias:
        #Los datos de entrada serán una tupla con tensores de pytorch, pasamos cada una:
        print(f'Ejecutando secuencia {j}')
        j+=1
        inputs={'A':i.T}
        network.run(inputs=inputs, time=T)
        
        #Obtenemos los spikes a lo largo de la simulación:
        spikes = {
            "X": source_monitor.get("s"), "B": target_monitor.get("s")
        }
        sp0.append(spikes['X'].sum(axis=2))
        sp1.append(spikes['B'].sum(axis=2))
        voltages = {"Y": target_monitor.get("v")}
    
    #Concatenamos y devolvemos:
    sp0=torch.concatenate(sp0)
    sp0=sp0.detach().numpy()
    
    sp1=torch.concatenate(sp1)
    sp1=sp1.detach().numpy()
    
    return [sp0,sp1,network]


# Create the network.
network, source_monitor,target_monitor = crear_red()

#Input folders
#Probar con dos ejecuciones de spikes bien, a ver qué pasa.
secuencias_bien2train=leer_data('train.csv',T)
secuencias2test=leer_data('test.csv',T)

print(f'Longitud de dataset de entrenamiento: {len(secuencias_bien2train)}')
# Simulate network on input data.
spikes_input,spikes,network=ejecutar_red(secuencias_bien2train,network,source_monitor,target_monitor,T)

#Establecemos que estamos de inferencia:
network.learning=False

print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
spikes_input,spikes,network=ejecutar_red(secuencias2test,network,source_monitor,target_monitor,T)

np.savetxt('spikes',spikes,delimiter=',')

with open('n1','w') as n1:
    n1.write(f'{R}\n')

with open('n2','w') as n2:
    n2.write(f'{n}\n')
