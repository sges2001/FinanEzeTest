''' Pair trading  con todos los pares
    utilizando volumen para pesar los retornos como hacen Avellaneda & Lee (2010) ecuacion 20

'''
import numpy as np, os, copy
import matplotlib.pyplot as plt
from time import time
from read_data import load_ts
import arbitrage as ar
import plotting as gra
import cointegration as co
import utils

class cnf:
    pathdat = 'dat/'             
    tipo = 'asset'              
    mtd = 'on'                  
    Ntraining = 131            
    Njump = 70                   
    beta_win = 121           
    zscore_win = 41            
    sigma_co = 1.5              
    sigma_ve = 0.2            
    nmax = -1               
    nsel = 100                   
    fname = f'tmp2/vol/'      
    linver_betaweight = 0  
    industry = 'beverages'       
    shorten = 0                  

os.makedirs(cnf.fname, exist_ok=True)
print("directorio ok")

## cargo series de tiempo: precios, volumen, compañías, etc.
day, date, price, company, volume = load_ts(sector=cnf.industry, pathdat=cnf.pathdat)
nt = price.shape[1]  

iini = 0 
caps_vol = [[] for _ in range(3)]     ## Capitales ajustadas por volumen
caps_novol = [[] for _ in range(3)]   ## Capitales sin ajuste
ratios = []                           ## voy a  ratios de volumen aplicados

for ilast in range(cnf.Ntraining + cnf.Njump, nt, cnf.Njump):
    print(iini, ilast, ilast - iini)

    ## Recorte de precios y volumen para esta ventana
    assets_tr = price[:cnf.nmax, iini:ilast]
    volume_tr = volume[:cnf.nmax, iini:ilast]

    t0 = time()
    res = ar.all_pairs(assets_tr, company[:cnf.nmax], cnf)
    print("Tiempo cálculo pares:", time() - t0)

    ## hago una copia para mantener una versión sin ajuste por volumen
    res_novol = copy.deepcopy(res)

    ## Ordenamos por half life
    metrics = co.all_pairs_stats(assets_tr[:, :ilast - cnf.Njump], company, 'asset')
    idx = np.argsort(metrics.half_life)[:cnf.nsel]  # Selección de mejores pares
    res.reorder(idx)
    res_novol.reorder(idx)

    ## Aplico el ajuste de volumen como en Avellaneda & Lee (2010)
    for par in range(21):  ## Selecciono los mejores 20 pares
        comp0 = res.company[par, 0]
        comp1 = res.company[par, 1]

        ix0 = np.where(company == comp0)[0][0]
        ix1 = np.where(company == comp1)[0][0]

        ## Extraigo los volumenes de los activos correspondientes a los mejores 20 pares segun hl
        vol0 = volume[ix0, iini:ilast]
        vol1 = volume[ix1, iini:ilast]

        ##volumen promedio teórico del par
        mean_vol_teo = (np.sum(vol0[:-cnf.Njump]) + np.sum(vol1[:-cnf.Njump])) / (2 * (ilast - iini - cnf.Njump))

        ## retornos por volumen actual
        for dia in range(res.retorno.shape[1]):
            v_actual = (vol0[dia] + vol1[dia]) / 2
            if v_actual > 100:  ## evitamos divisiones por valores pequeños o cero
                ratio = mean_vol_teo / v_actual
                ratios.append(ratio)
                res.retorno[par, dia] *= ratio
            else:
                res.retorno[par, dia] = 0  # Si no hay volumen, no hay retorno

    ## guarda retorno medio para distintos grupos (top 5, 10, 20 pares)
    caps_vol[0].append(res.retorno[:5, cnf.Ntraining:].mean(0))
    caps_vol[1].append(res.retorno[:10, cnf.Ntraining:].mean(0))
    caps_vol[2].append(res.retorno[:20, cnf.Ntraining:].mean(0))

    caps_novol[0].append(res_novol.retorno[:5, cnf.Ntraining:].mean(0))
    caps_novol[1].append(res_novol.retorno[:10, cnf.Ntraining:].mean(0))
    caps_novol[2].append(res_novol.retorno[:20, cnf.Ntraining:].mean(0))

    ## la ventana se mueve
    iini += cnf.Njump

## guardo los ratios de volumen
np.savetxt(cnf.fname + "ratios.csv", ratios, delimiter=",")

## concateno retornos y calculo evolución del capital
rets = np.array([np.concatenate(c) for c in caps_vol + caps_novol])
capitales = np.zeros_like(rets)
capitales[:, 0] = 100  # Capital inicial

for i in range(6):
    capitales[i, 1:] = capitales[i, 0] * np.cumprod(1 + rets[i, 1:])

# Grafico de evolución de capital ajustado vs no ajustado por volumen
figfile = cnf.fname + 'comparacion_vol_novol.png'
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
labels = ['VolAdj half 5', 'VolAdj half 10', 'VolAdj half 20',
          'NoVol half 5', 'NoVol half 10', 'NoVol half 20']

for i in range(6):
    ax.plot(capitales[i], label=labels[i])
#
ax.set(title='Comparación con y sin ajuste por volumen')
ax.legend()
plt.tight_layout()
fig.savefig(figfile)
plt.close()
