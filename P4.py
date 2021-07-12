#4.1 Modulación 16-QAM
from PIL import Image
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from scipy import fft

# Se copian las funciones necesarias dadas

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)


def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)


#moduladorI
def moduladorI(bits, fc, mpp):
    
    '''
    Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    bitsr = bits.reshape(len(bits)//2,2) 
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bitsr) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora en fase "I".
    Tc = 1 / fc  # Tiempo de un periodo de la portadora I.
    t_periodo_I = np.linspace(0, Tc, mpp)
    portadora_I = np.cos(2*np.pi*fc*t_periodo_I)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senalI = np.zeros(t_simulacion.shape)
    moduladoraI = np.zeros(t_simulacion.shape)  # señal de información
 
    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(bitsr):
        if bitsr[i,0] == 1:
            senalI[i*mpp : (i+1)*mpp] = portadora_I
            moduladoraI[i*mpp : (i+1)*mpp] = 1
        else:
            senalI[i*mpp : (i+1)*mpp] = portadora_I * -1
            moduladoraI[i*mpp : (i+1)*mpp] = 0
    
    return senalI, portadora_I, moduladoraI, t_simulacion, Tc, N  

#ModuladorQ
def moduladorQ(bits, fc, mpp):
    
    bitsr = bits.reshape(len(bits)//2,2)
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bitsr) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora "Q".
    Tc = 1 / fc  # Tiempo de un periodo de la portadora Q.
    t_periodoQ = np.linspace(0, Tc, mpp)
    portadoraQ = np.sin(2*np.pi*fc*t_periodoQ)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senalQ = np.zeros(t_simulacion.shape)
    moduladoraQ = np.zeros(t_simulacion.shape)  # señal de información
 
    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(bitsr):
        if bitsr[i,1] == 1:
            senalQ[i*mpp : (i+1)*mpp] = portadoraQ
            moduladoraQ[i*mpp : (i+1)*mpp] = 1
        else:
            senalQ[i*mpp : (i+1)*mpp] = portadoraQ * -1
            moduladoraQ[i*mpp : (i+1)*mpp] = 0


    
    return senalQ, portadoraQ, moduladoraQ 
    
# Señal modulada
def senalModulada(senalI, senalQ, moduladoraI, moduladoraQ, tsimulacion, Tc, N):
    
    # Se realiza la suma de las señales provenientes de cada portadora. 
    senalTx = senalI + senalQ
    
    # Se realiza la suma de las moduladoras provenientes de cada portadora.
    moduladora = moduladoraI + moduladoraQ
    
    
    # Calcular la potencia promedio de la señal modulada
    Pm = (1 / (N*Tc)) * np.trapz(pow(senalTx, 2), tsimulacion)
    
    return senalTx, Pm, moduladora

#Canal AWGN
def canal_ruidoso(senalTx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senalTx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senalTx + ruido

    return senal_Rx


#demodulación

# Se define la función "demodulador"
def demodulador(senal_Rx, portadoraI, portadoraQ, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema QPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora_I: La onda portadora "en fase"
    :param portadora_Q: La onda portadora "en cuadratura"
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bitsRx_I = np.zeros(N)
    bitsRx_Q = np.zeros(N)
    bits_Rx = np.zeros(2*N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(M)

    # Pseudo-energía de un período de la portadora "I"
    Es_I = np.sum(portadoraI**2)
    
    # Pseudo-energía de un período de la portadora "Q"
    Es_Q = np.sum(portadoraQ**2)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        productoI = senal_Rx[i*mpp : (i+1)*mpp] * portadoraI
        productoQ = senal_Rx[i*mpp : (i+1)*mpp] * portadoraQ
        Ep_I = np.sum(productoI)
        Ep_Q = np.sum(productoQ)
        senal_demodulada[i*mpp : (i+1)*mpp] = productoI + productoQ
         

        # Criterio de decisión por detección de energía
        if Ep_I > 0:
            bitsRx_I[i] = 1
        else:
            bitsRx_I[i] = 0
            
        if Ep_Q > 0:
            bitsRx_Q[i] = 1
        else:
            bitsRx_Q[i] = 0
            
    # Se vuelven a acomodar los bits en el orden apropiado.        
    for j, bits_I in enumerate(bitsRx_I):
        bits_Rx[2*j] = bits_I 
        
    for k, bitsQ in enumerate(bitsRx_Q):
        bits_Rx[2*k+1] = bitsQ

    return bits_Rx.astype(int), senal_demodulada


#Reconstrucción de la imagen

def bits_a_rgb(bitsRx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


#simulacion


# Parámetros
fc = 5000  # frecuencia de ambas portadoras
mpp = 20   # muestras por periodo de la portadora
SNR = -5    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema QAM
senal_I, portadora_I, moduladora_I, t_simulacion, Tc, N = moduladorI(bits_Tx, fc, mpp)

# 4. Modular la cadena de bits usando el esquema QAM
senal_Q, portadora_Q, moduladora_Q = moduladorQ(bits_Tx, fc, mpp)

# 5. Modular la cadena de bits usando el esquema QAM
senal_Tx, Pm, moduladora = senalModulada(senal_I, senal_Q, moduladora_I, moduladora_Q, t_simulacion, Tc, N)

# 6. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 7. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora_I, portadora_Q, mpp)

# 8. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 8. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Enviado')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recibido')
Fig.tight_layout()

plt.imshow(imagen_Rx)


# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

# La señal modulada por BPSK
ax1.plot(senal_Tx[0:600], color='g', lw=2) 
ax1.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax2.plot(senal_Rx[0:600], color='b', lw=2) 
ax2.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax3.plot(senal_demodulada[0:600], color='m', lw=2) 
ax3.set_ylabel('$b^{\prime}(t)$')
ax3.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()




#--------------------------------------------------------------------------------------------------------




#4.2 Estacionaridad y Ergodicidad

# Frecuencia de Portadoras
fc = 5000

# Variables aleatorias A1 y A2
vaA1 = stats.bernoulli(1/2)
vaA2 = stats.bernoulli(1/2) 

# Vector de tiempo 
T = 100    # Número de elementos
tf = 10    # Tiempo final
t = np.linspace(0, tf, T)

# Función del tiempo S
n = 10000
St = np.empty((n, len(t))) 

# Muestras
for i in range(n):
    a1 = vaA1.rvs()
    a2 = vaA2.rvs()
    if a1==0:
        a1=-1
    else:
        a1=1 
    if a2==0:
        a2=-1
    else:
        a2=1
    st = a1 * np.cos(2*np.pi*fc*t) + a2 * np.sin(2*np.pi*fc*t)
    St[i,:] = st
    plt.plot(t, st)
    
# Promedio P
P = [np.mean(St[i,:]) for i in range(len(t))]
plt.plot(t, P, lw=3, label='Valor teórico')

# Promedio se la senal_Tx.
P = [np.mean(senal_Tx) for i in range(len(t))]
plt.plot(t, P, '-.', lw=3, label='Valor de la señal modulada')

plt.title('Realizaciones del proceso aleatorio $s(t)$')
plt.xlabel('$t$')
plt.ylabel('$s(t)$')
plt.legend()
plt.show() 



#--------------------------------------------------------------------------------------------------------

'''

# Densidad espectral de potencia


# Transformada de Fourier 
senal_f = fft(senal_Tx) # La función fft realiza el cálculo de la transformada rápida de Fourier de la senal_Tx. 

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos
Ns = Nm // mpp

# Tiempo del símbolo
Ts = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Ts / mpp

# Tiempo de la simulación
T = Ns * Ts

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Se define la densidad espectral de potencia Sxx = |s(w)|^2
S_xx = np.power(np.abs(senal_f), 2)

# Densidad espectral de potencia
print('Densidad espectral de potencia para la señal modulada\n')
print(S_xx)

# Densidad espectral de potencia 
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2), color = 'greem' , label='$Sxx(f)$')
plt.xlim(0, 20000)
plt.title('Densidad espectral de Potencia vs Frencuencia')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia')
plt.legend()
plt.grid()   
plt.show() 
'''